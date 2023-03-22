import torch
import numpy as np
import random
from sklearn.model_selection import KFold
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch.distributions import normal
import argparse
from scipy import io
from torch.distributions import normal, kl
from plot import plot, plot_matrix
import matplotlib.pyplot as plt
from model_rbgm import GNN_1,frobenious_distance
import timeit
from data_utils import timer
import copy

# random seed
manualSeed = 1

np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('running on GPU')
    # if you are using GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

else:
    device = torch.device("cpu")
    print('running on CPU')


def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-mode', type=str, default="4D-FED-GNN++", help='training technique')
    parser.add_argument('-num_folds', type=int, default=5, help='cv number')
    parser.add_argument('--num_regions', type=int, default=35,
                        help='Number of regions')
    parser.add_argument('--num_timepoints', type=int, default=3,
                        help='Number of timepoints')
    parser.add_argument('-num_epochs', type=int, default=60, help='number of epochs')
    parser.add_argument('--lr_g', type=float, default=0.01, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=0.0002, help='Discriminator learning rate')
    parser.add_argument('--decay', type=float, default=0.0, help='Weight Decay')
    parser.add_argument('-C', type=int, default=14, help='number of round before averaging')
    parser.add_argument('-D', type=int, default=7, help='number of rounds before daisy chain')
    parser.add_argument('-batch_num', type=int, default=1, help='batch number')
    parser.add_argument('--tp_coeff', type=float, default=0.0, help='Coefficient of topology loss')
    parser.add_argument('--g_coeff', type=float, default=2.0, help='Coefficient of adversarial loss')
    parser.add_argument('--i_coeff', type=float, default=2.0, help='Coefficient of identity loss')
    parser.add_argument('--kl_coeff', type=float, default=0.001, help='Coefficient of KL loss')
    parser.add_argument('--exp', type=int, default=1, help='Which experiment are you running')
    parser.add_argument('--lr', type=float, default=0.001, help="Learninng rate")
    parser.add_argument('--tp_coef', type=float, default=10, help="KL Loss Coefficient")
    args, _ = parser.parse_known_args()
    return args


def create_edge_index_attribute(adj_matrix):
    """
    Given an adjacency matrix, this function creates the edge index and edge attribute matrix
    suitable to graph representation in PyTorch Geometric.
    """

    rows, cols = adj_matrix.shape[0], adj_matrix.shape[1]
    edge_index = torch.zeros((2, rows * cols), dtype=torch.long)
    edge_attr = torch.zeros((rows * cols, 1), dtype=torch.float)
    counter = 0

    for src, attrs in enumerate(adj_matrix):
        for dest, attr in enumerate(attrs):
            edge_index[0][counter], edge_index[1][counter] = src, dest
            edge_attr[counter] = attr
            counter += 1

    return edge_index, edge_attr, rows, cols


class Hospital():
    def __init__(self, args):
        """
        Hospital object contains a GNN and an optimizer for each timepoint

        Hospital object can update GNN-layer wise weights of its GNNs
        """

        self.models = []
        self.optimizers = []
        for i in range(args.num_timepoints - 1):
            self.models.append(GNN_1().to(device))
            self.optimizers.append(torch.optim.Adam(self.models[i].parameters(), lr=args.lr))

    def update_hospital(self, main_model):
        for i in range(len(self.models)):
            self.models[i].load_state_dict(main_model.models[i].state_dict())


def get_folds(length, num_folds):
    """
    Arguments:
        length: number of subjects
        num_folds: number of folds

    This function returns a list of subjects for each fold (list of lists)
    """
    indexes = list(range(length))
    random.shuffle(indexes)
    n = length // num_folds

    folds = []
    for fold in range(num_folds):
        if fold == num_folds - 1:
            folds.append(indexes[fold * n: -1])
        else:
            folds.append(indexes[fold * n: (fold * n) + n])

    return folds


def get_order(table):
    """
    Computes the order of the hospitals
    A hospital score is calculated as: number of 1s + the last timepoint availability point
    """
    sums = np.sum(table, axis=1)

    sums += table[:, -1]

    order = np.argsort(sums)
    order = np.flip(order)

    return order


def train(args, dataset, table):
    """
        Arguments:
            args: arguments
            dataset: the whole dataset (train and test set)
            table: [num_hospitals, num_timepoints], holds timepoint-wise availability of hospitals

        This function performs training and testing reporting Mean Absolute Error (MAE) of the testing brain graphs.
        """
    folds = get_folds(dataset.shape[0], args.num_folds)
    indexes = range(args.num_folds)
    kfold = KFold(n_splits=args.num_folds)
    f = 0
    for train, test in kfold.split(indexes):
        tic0 = timeit.default_timer()
        print(
            f'------------------------------------Fold [{f + 1}/{args.num_folds}]-----------------------------------------')
        # initialize hospitals
        hospitals = []
        mae_list, tp_list, tot_list = list(), list(), list()
        train_data_list = []
        for h in range(args.num_folds - 1):
            hospitals.append(Hospital(args))
            train_data_list.append(dataset[folds[train[h]]])

        # start training
        for t in range(1, args.num_timepoints):
            print("-----------------------------------------------------------------------------")
            # determine the order
            ordered_hospitals = get_order(table[:, t - 1:t + 1])
            print("Ordering of the hospitals: ", ordered_hospitals)
            for epoch in range(args.num_epochs):
                print(f'Epoch [{epoch + 1}/{args.num_epochs}]')
                tot_mae, tot, tp = 0.0, 0.0, 0.0
                for h_i in ordered_hospitals:
                    h = hospitals[h_i]
                    train_data = train_data_list[h_i]
                    print(f'Hospital [{h_i + 1}/{len(hospitals)}]')
                    hospitals[h_i], tot_l, tp_l, mae_l = train_one_epoch(args, h, train_data, f, table, [h_i, t])
                    tot_mae += mae_l
                    tot += tot_l
                    tp += tp_l

                    print(f'[Train] Loss T' + str(t) + f': {mae_l:.5f}',
                          f'[Train] TP Loss T' + str(t) + f': {tp_l:.5f} ',
                          f'[Train] Total Loss T' + str(t) + f': {tot_l:.5f} ')

                if epoch != args.num_epochs - 1 or epoch != 0:
                    if epoch % args.C == 0 and args.mode != "4D-GNN":
                        hospitals = update_main_by_average(hospitals, t)
                    elif epoch % args.D == 0 and args.mode == "4D-FED-GNN+":
                        hospitals = exchange_models(hospitals, t)
                    elif epoch % args.D == 0 and args.mode == "4D-FED-GNN++":
                        hospitals = exchange_models_based_on_order(hospitals, t, ordered_hospitals)

                mae_list.append(tot_mae)
                tot_list.append(tot)
                tp_list.append(tp)

            plot("Total loss", "model" + str(t) + "totalLossTrainSet" + str(f) + "_exp" + str(args.exp), tot_list)
            plot("MAE", "model" + str(t) + "MAELossTrainSet" + str(f) + "_exp" + str(args.exp), mae_list)
            plot("TP", "model" + str(t) + "tpLossTrainSet" + str(f) + "_exp" + str(args.exp), tp_list)
            print(" ")
        test_data = dataset[folds[f][0]]
        validate(args, hospitals, test_data, f)
        tic1 = timeit.default_timer()
        timer(tic0,tic1)


def exchange_models(hospitals, t):
    """
        This function exchanges GNNs of hospitals at timepoint t with each other
    """
    pre_model = None
    for i, hospital in enumerate(hospitals):
        next_model = copy.deepcopy(hospitals[i].models[t - 1].state_dict())

        if not pre_model is None:
            hospitals[i].models[t - 1].load_state_dict(pre_model)

        pre_model = copy.deepcopy(next_model)

        if i == 0:
            hospitals[i].models[t - 1].load_state_dict(copy.deepcopy(hospitals[-1].models[t - 1].state_dict()))

    return hospitals


def exchange_models_based_on_order(hospitals, t, order):
    """
        This function exchanges GNN-layer weights of paired hospitals at timepoint t with each other
    """
    pre_model = None
    for i, h_i in enumerate(order):
        next_model = copy.deepcopy(hospitals[h_i].models[t - 1].state_dict())

        if not pre_model is None:
            hospitals[h_i].models[t - 1].load_state_dict(pre_model)

        pre_model = copy.deepcopy(next_model)

        if i == 0:
            hospitals[h_i].models[t - 1].load_state_dict(copy.deepcopy(hospitals[order[-1]].models[t - 1].state_dict()))

    return hospitals


def validate(args, hospitals, test_data, f):
    """
        Output:
            plotting of each predicted testing brain graph, also saved as a numpy file
            average MAE of predicted brain graphs
    """
    mael = torch.nn.L1Loss().to(device)

    val_hos = len(test_data)
    for j, hospital in enumerate(hospitals):
        hloss = []
        for k in range(len(hospital.models)):
            hospital.models[k].eval()
            hloss.append(0)

        with torch.no_grad():
            for i, data in enumerate(test_data):
                data = data.to(device)
                out_1 = data[0]
                for k, model in enumerate(hospital.models):
                    temp = model.rnn[0].hidden_state
                    out_1 = model(out_1)
                    model.rnn[0].hidden_state = temp
                    hloss[k] += mael(out_1, data[k + 1])
                    plot_matrix(data[k].cpu().detach().numpy(), "t" + str(k) + "gt" + str(i))
                    plot_matrix(out_1.cpu().detach().numpy(), "exp_" + str(args.exp) + "t" + str(k + 1) + "_sample" + str(i) + "_hos" + str(j))
                    np.save("np_graphs/t" + str(k) + "gt" + str(i), data[k].cpu().detach().numpy())
                    np.save("np_graphs/exp_" + str(args.exp) + "t" + str(k + 1) + "_sample" + str(i) + "_hos" + str(j), out_1.cpu().detach().numpy())

                plot_matrix(data[-1].cpu().detach().numpy(), "t" + str(k) + "gt" + str(i))
                np.save("np_graphs/t" + str(k + 1) + "gt" + str(i), data[-1].cpu().detach().numpy())

        for k in range(1, args.num_timepoints):
            print(
                '[Val]: MAE Loss Model' + str(k) + f': {hloss[k - 1] / val_hos:.5f}', sep=' ', end='', flush=True)

        print(" ")


def update_main_by_average(hospitals, t):
    """
        This function takes the GNN-layer weights of the GNN at timepoint t and computes the global model by averaging,
        then broadcats the weights to the hospitals (updates each GNN with the global model)
    """
    target_state_dict = copy.deepcopy(hospitals[0].models[t - 1].state_dict())
    mux = 1 / len(hospitals)

    model_state_dict_list = [copy.deepcopy(hospitals[i].models[t - 1].state_dict()) for i in range(1, len(hospitals))]

    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data = target_state_dict[key].data.clone() * mux
            for model_state_dict in model_state_dict_list:
                target_state_dict[key].data += mux * model_state_dict[key].data.clone()

    for i in range(len(hospitals)):
        hospitals[i].models[t - 1].load_state_dict(target_state_dict)

    return hospitals



def train_one_epoch(args, hospital, train_data, fold, table, index):
    """
    Arguments:
        hospital: the currently training hospital
        train_data: local data of the hospital
        table: the table that holds the timepoint-wise availability of the hospitals
        index: [hospital_id, timepoint]

    This function trains the GNN of hospital-of-interest for one epoch based on its availability at current and next timepoints.
    If the next timepoint is available:
        supervised learning
    else:
        self-learning

    Returns:
        hospital, total loss, topological loss, mae loss
    """
    mael = torch.nn.L1Loss().to(device)
    tp = torch.nn.MSELoss().to(device)

    total_step = len(train_data)
    train_loss = 0.0
    tp_loss, tr_loss = 0.0, 0.0

    cur_id = index[1] - 1  # id of the model that will be trained
    if table[index[0], index[1] - 1] == 0 and table[index[0], index[1]] == 0:
        # if the hospital doesn't have data at both current and next timepoints
        gt = 0
        for j in reversed(range(cur_id)):
            # find the first GNN that has its follow-up data
            if table[index[0], j] == 1:
                gt = j
                break
        hospital.models[cur_id].train()
        for i, x in enumerate(train_data):
            # take the ground-truth data and keep predicting until reaching the current timepoint
            data = x[gt]
            with torch.no_grad():
                for j in range(gt, cur_id):
                    data = data.to(device)
                    data = hospital.models[j](data)

            # predict the brain graph for the next timepoint
            hospital.optimizers[cur_id].zero_grad()
            out = hospital.models[cur_id](data)

            # Topological Loss
            tp_l = tp(out.sum(dim=-1), data.sum(dim=-1))
            tp_loss += tp_l.item()

            # MAE Loss
            loss = mael(out, data)
            train_loss += loss.item()

            total_loss = loss + args.tp_coef * tp_l
            tr_loss += total_loss.item()
            total_loss.backward()
            hospital.optimizers[cur_id].step()
    if table[index[0], index[1] - 1] == 1 and table[index[0], index[1]] == 0:
        # if the hospital doesn't have data at next timepoint
        hospital.models[cur_id].train()
        for i, data in enumerate(train_data):
            data = data.to(device)
            hospital.optimizers[cur_id].zero_grad()
            out = hospital.models[cur_id](data[cur_id])

            # self-learning

            # Topological Loss
            tp_l = tp(out.sum(dim=-1), data[cur_id].sum(dim=-1))
            tp_loss += tp_l.item()

            # MAE Loss
            loss = mael(out, data[cur_id])
            train_loss += loss.item()

            total_loss = loss + args.tp_coef * tp_l
            tr_loss += total_loss.item()
            total_loss.backward()
            hospital.optimizers[cur_id].step()

    elif table[index[0], index[1] - 1] == 0 and table[index[0], index[1]] == 1:
        # if the hospital doesn't have data at current timepoint
        gt = 0
        for j in reversed(range(cur_id)):
            # find the first GNN that has its follow-up data
            if table[index[0], j] == 1:
                gt = j
                break
        hospital.models[cur_id].train()
        for i, x in enumerate(train_data):
            # take the ground-truth data and keep predicting until reaching the current timepoint
            data = x[gt]
            with torch.no_grad():
                for j in range(gt, cur_id):
                    data = data.to(device)
                    data = hospital.models[j](data)

            hospital.optimizers[cur_id].zero_grad()
            out = hospital.models[cur_id](data)

            # Topological Loss
            tp_l = tp(out.sum(dim=-1), x[cur_id + 1].sum(dim=-1))
            tp_loss += tp_l.item()

            # MAE Loss
            loss = mael(out, x[cur_id + 1])
            train_loss += loss.item()

            # self-learning loss

            # Topological Loss
            self_tp_l = tp(out.sum(dim=-1), data.sum(dim=-1))
            tp_loss += self_tp_l.item()

            # MAE Loss
            self_loss = mael(out, data)
            train_loss += self_loss.item()

            total_loss = (loss + self_loss + args.tp_coef * tp_l + args.tp_coef * self_tp_l) / 2
            tr_loss += total_loss.item()
            total_loss.backward()
            hospital.optimizers[cur_id].step()
    elif table[index[0], index[1] - 1] == 1 and table[index[0], index[1]] == 1:
        # if the hospital have data at both timepoints
        hospital.models[cur_id].train()
        for i, data in enumerate(train_data):
            data = data.to(device)
            hospital.optimizers[cur_id].zero_grad()
            out = hospital.models[cur_id](data[cur_id])

            # Topological Loss
            tp_l = tp(out.sum(dim=-1), data[cur_id + 1].sum(dim=-1))
            tp_loss += tp_l.item()

            # MAE Loss
            loss = mael(out, data[cur_id + 1])
            train_loss += loss.item()

            # self-learning loss

            # Topological Loss
            self_tp_l = tp(out.sum(dim=-1), data[cur_id].sum(dim=-1))
            tp_loss += self_tp_l.item()

            # MAE Loss
            self_loss = mael(out, data[cur_id])
            train_loss += self_loss.item()

            total_loss = (loss + self_loss + args.tp_coef * tp_l + args.tp_coef * self_tp_l) / 2
            tr_loss += total_loss.item()
            total_loss.backward()
            hospital.optimizers[cur_id].step()

    tot = tr_loss / total_step
    tp_l = tp_loss / total_step
    mae = train_loss / total_step

    for i, model in enumerate(hospital.models):
        # Save the models
        torch.save(model.state_dict(),
                   "./weights/hos" + str(i + 1) + "model" + str(index[1]) + "_" + str(fold) + "_exp" + str(args.exp) + ".model")

    return hospital, tot, tp_l, mae


def random_table(args, size):
    """
        Returns a table where each slot is randomly filled with zero or one based on a ratio
    """

    table = np.ones((args.num_folds - 1, args.num_timepoints))
    comb = np.zeros((args.num_folds - 1) * (args.num_timepoints - 1))
    comb[: int(size * comb.shape[0])] = 1
    for i in range(1):
        np.random.shuffle(comb)
    comb = comb.reshape(args.num_folds - 1, args.num_timepoints - 1)
    table[:, 1:] = comb
    return table


if __name__ == "__main__":
    args = get_args()
    dataset = np.load("multivariate_simulation_data.npy")
    dataset = torch.from_numpy(dataset)
    dataset = dataset.type(torch.FloatTensor)

    table = np.zeros((args.num_folds - 1, args.num_timepoints))
    table = random_table(args, 4/8)
    print(table)
    train(args, dataset[:, :, :, :, 0], table)
