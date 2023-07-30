import os
import torch
import numpy as np
import random
import time
from sklearn.model_selection import KFold
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch.distributions import normal
import argparse
from scipy import io
import seaborn as sns
from torch.distributions import normal, kl
from plot import plot, plot_matrix
import matplotlib.pyplot as plt
from model_rbgm import GNN_1,frobenious_distance
import timeit
from data_utils import timer
import copy
#from lion_pytorch import Lion

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
    parser.add_argument('-mode', type=str, default="weighted_weight_exchange", help='training technique')
    parser.add_argument('-num_folds', type=int, default=5, help='cv number')
    parser.add_argument('-num_hospitals', type=int, default=4, help='hospitals')
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
    parser.add_argument('-save_path',type=str,default = '/Users/pavelbozmarov/Desktop/Python_Projects/Imperial/Dissertation/Code/4D-FedGNN-Plus_mine/results/',help='Path to the saved results')
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
            #self.optimizers.append(Lion(self.models[i].parameters(), lr=args.lr))

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


def get_order_original(table):
    """
    Computes the order of the hospitals
    A hospital score is calculated as: number of 1s + the last timepoint availability point
    """
    sums = np.sum(table, axis=1)

    sums += table[:, -1]

    order = np.argsort(sums)
    order = np.flip(order)

    return order
    
def get_order_weighted(table):
    """
    Computes the order of the hospitals
    A hospital score is calculated as: number of 1s + the last timepoint availability point
    """
    sums = np.sum(table, axis=1).astype(int)
    # Get sorted indices in descending order
    sorted_indices = np.argsort(sums)[::-1]
    # Create list with sorted indices and items
    result = [[index, sums[index]] for index in sorted_indices]
    
    return result
    


def train(args, dataset, table,verbose,train_validate_verbose,train_validate_verbosity_epochs):
    """
        Arguments:
            args: arguments
            dataset: the whole dataset (train and test set)
            table: [num_hospitals, num_timepoints], holds timepoint-wise availability of hospitals

        This function performs training and testing reporting Mean Absolute Error (MAE) of the testing brain graphs.
        """
    
    # Create the results folders
    print('hi')
    os.makedirs(args.save_path+'results', exist_ok=True)
    os.makedirs(args.save_path+f'results/{args.save_name}', exist_ok=True)
    os.makedirs(args.save_path+f'results/{args.save_name}/real_and_predicted_graphs', exist_ok=True)
    os.makedirs(args.save_path+f'results/{args.save_name}/train_losses', exist_ok=True)
    os.makedirs(args.save_path+f'results/{args.save_name}/train_losses/mae_losses', exist_ok=True)
    os.makedirs(args.save_path+f'results/{args.save_name}/train_losses/tp_losses', exist_ok=True)
    os.makedirs(args.save_path+f'results/{args.save_name}/train_losses/total_losses', exist_ok=True)
    os.makedirs(args.save_path+f'results/{args.save_name}/test_mae_losses', exist_ok=True)
    os.makedirs(args.save_path+f'results/{args.save_name}/trained_models', exist_ok=True)
    
    # Create the results folders
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
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # !!! necessary for the below line
        torch.use_deterministic_algorithms(True)
        print('TRAIN deterministic algorithms')

    else:
        device = torch.device("cpu")
        print('running on CPU')

    # change the save path
    args.save_path = args.save_path+f'results/{args.save_name}/'

    # Choosing the right get_order function
    if args.mode == 'weighted_weight_exchange':
        get_order =  get_order_weighted
    else:
        get_order = get_order_original
    
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
        train_data_list = []
        for h in range(args.num_folds - 1):
            hospitals.append(Hospital(args))
            train_data_list.append(dataset[folds[train[h]]])

        # start training
        for t in range(1, args.num_timepoints):
            print('time point:',t)
            mae_list, tp_list, tot_list = list(), list(), list()
            if verbose:
                print("-----------------------------------------------------------------------------")
            
            # determine the order
            ordered_hospitals = get_order(table[:, t - 1:t + 1])
           
            if verbose:
                print("Ordering of the hospitals: ", ordered_hospitals)
            epochs_start = time.time()
            for epoch in range(args.num_epochs):
                if verbose:
                    print(f'Epoch [{epoch + 1}/{args.num_epochs}]')
                tot_mae, tot, tp = 0.0, 0.0, 0.0
                for item in ordered_hospitals:
                    
                    if args.mode == 'weighted_weight_exchange':
                        h_i,w_i = item
                    else:
                        h_i = item
                        
                    h = hospitals[h_i]
                    train_data = train_data_list[h_i]
                    if verbose:
                        print(f'Hospital [{h_i + 1}/{len(hospitals)}]')
                    hospitals[h_i], tot_l, tp_l, mae_l = train_one_epoch(args, h, train_data, f, table, [h_i, t])
                    tot_mae += mae_l
                    tot += tot_l
                    tp += tp_l
                    if verbose:
                        print(f'[Train] Loss T' + str(t) + f': {mae_l:.5f}',
                          f'[Train] TP Loss T' + str(t) + f': {tp_l:.5f} ',
                          f'[Train] Total Loss T' + str(t) + f': {tot_l:.5f} ')
                if train_validate_verbose and (epoch+1)%train_validate_verbosity_epochs==0 :
                    
                    test_data = dataset[folds[test[0]]]
                    val_loss = validate_during_training(args, hospitals, test_data,t)
                    print(f"Epoch:{epoch+1},val_loss:{val_loss}")
                        
                if epoch != args.num_epochs - 1 or epoch != 0:
                    if epoch % args.C == 0 and args.mode != "4D-GNN":
                        hospitals = update_main_by_average(hospitals, t)
                    elif epoch % args.D == 0 and args.mode == "4D-FED-GNN+":
                        hospitals = exchange_models(hospitals, t)
                    elif epoch % args.D == 0 and args.mode == "4D-FED-GNN++":
                        hospitals = exchange_models_based_on_order(hospitals, t, ordered_hospitals)
                    elif epoch % args.D == 0 and args.mode == "weighted_weight_exchange":
                        hospitals = exchange_models_weights_pairs_extreme(hospitals, t, ordered_hospitals)

                mae_list.append(tot_mae)
                tot_list.append(tot)
                tp_list.append(tp)

            plot(tot_list,f"Total Train Loss  Model {t} Fold {f}",'Total Loss',args.save_path+'train_losses/total_losses/'+ f"total_train_loss_model_{t}_fold_{f}", verbose)
            plot(mae_list,f"MAE Loss Model {t} Fold {f}",'MAE Loss', args.save_path+'train_losses/mae_losses/'+ f"mae_train_loss_model_{t}_fold_{f}", verbose)
            plot(tp_list,f"TP Loss Model {t} Fold {f}", 'TP Loss', args.save_path+'train_losses/mae_losses/'+ f"tp_train_loss_model_{t}_fold_{f}", verbose)

            if verbose:
                print(" ")
        
        epochs_end = time.time()-epochs_start
        print(f'epochs finished with time:{epochs_end}')
        test_data = dataset[folds[test[0]]]
        validate(args, hospitals, test_data, f,verbose)
        tic1 = timeit.default_timer()
        timer(tic0,tic1)
        f+=1
    
    # save the weights for the hospitals
    for i,hospital in enumerate(hospitals):
        for j,model in enumerate(hospital.models):

            # save the model of the current hospital
            torch.save(model.state_dict(),
                   args.save_path +f'trained_models/hospital_{i}_model_{j+1}')



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
    
def exchange_models_weights_pairs_extreme(hospitals,t,order):
    
    """
        This function exchanges GNN-layer weights of paired hospitals at timepoint t with each other.
        We pair strong strongest hospitals with weakest ones.
    """
    
    if len(order)%2==1:
        
        h_last,w_last = order[-1]
        order = order[:-1]
    
    for i in range(int(len(order)/2)):
        h_strong,w_strong = order[i]
        h_weak,w_weak = order[len(order)-(i+1)]

         # Calculate the total weight for normalization
        total_weight = w_strong + w_weak

        # Initialize an empty state dictionary to hold the weighted average of the models
        avg_state_dict = {name: torch.zeros_like(param) for name, param in hospitals[h_strong].models[t - 1].state_dict().items()}
        
        # Calculate the weighted average of the strong and weak models
        for name, param_strong in hospitals[h_strong].models[t - 1].state_dict().items():
            
            param_weak = hospitals[h_weak].models[t - 1].state_dict()[name]
            avg_state_dict[name] = (w_strong * param_strong + w_weak * param_weak) / total_weight

        # Update both models in the pair with the weighted average
        hospitals[h_strong].models[t - 1].load_state_dict(avg_state_dict)
        hospitals[h_weak].models[t - 1].load_state_dict(avg_state_dict)
        

    return hospitals  


def validate(args, hospitals, test_data, f,verbose):
    """
        Output:
            plotting of each predicted testing brain graph, also saved as a numpy file
            average MAE of predicted brain graphs
    """
    mael = torch.nn.L1Loss().to(device)
    
    for j, hospital in enumerate(hospitals):
        hloss = []
        for k in range(len(hospital.models)):
            hospital.models[k].eval()
            hloss.append(torch.tensor(0))

        with torch.no_grad():
            for i, data in enumerate(test_data):
                data = data.to(device)
                out_1 = data[0] # 
                for k, model in enumerate(hospital.models):
                    temp = model.rnn[0].hidden_state
                    out_1 = model(out_1)
                    model.rnn[0].hidden_state = temp
                    
                    # Updating the MAE loss for hospital j, model k
                    hloss[k] = hloss[k] + mael(out_1, data[k + 1])
                    #print(f'MAE LOSS Hospital_{j}_Subject_{i}_Timepoint_{k+1}_Fold_{f}: {mael(out_1, data[k + 1])}')
                    # plot and save the brain graph of patient(sample) i through all the timepoints 
                    plot_matrix(data[k+1].cpu().detach().numpy(),f'Real Graph, Hospital:{j}, Subject:{i}, Timepoint:{k+1}',
                               args.save_path+'real_and_predicted_graphs/'+f'hospital_{j}_subject_{i}_timepoint_{k+1}_fold_{f}_real_graph',verbose)
                    plot_matrix(out_1.cpu().detach().numpy(),f'Predicted Graph, Hospital:{j}, Subject:{i}, Timepoint:{k+1}',
                               args.save_path+'real_and_predicted_graphs/'+f'hospital_{j}_subject_{i}_timepoint_{k+1}_fold_{f}_predicted_graph',verbose)
        
        print(F'OVERALL PER PIXEL MAE LOSS FOR HOSPITAL:{j} FOR ALL MODELS')
        print(np.array([item.cpu()/len(test_data) for item in hloss]))         
        # Save the MAE Loss
        np.save(args.save_path+f"test_mae_losses/mae_test_loss_hospital_{j}_fold_{f}", np.array([item.cpu()/len(test_data) for item in hloss]))
        
        if verbose:
            print(f'Hospital:{j}')
            for k in range(args.num_timepoints-1):
                
                print(
                    '[Val]: MAE Loss Model' + str(k) + f': {hloss[k] / len(test_data):.5f}', sep=' ', end='')

            print(" ")

def validate_during_training(args, hospitals, test_data,t):
    """
    This function calculates the average MAE of predicted brain graphs during validation.
    We only use the models that are related to data prediction for timepoint t. These are the 
    models with indices t-1
    """
    mael = torch.nn.L1Loss().to(device)
    val_hos = len(test_data)
    
    hloss=[]
    for i, hospital in enumerate(hospitals):
           hospital.models[t-1].eval()
           hloss.append(0)

           with torch.no_grad():
                for data in test_data:
                    data = data.to(device)
                    # here our input data is the data at timepoint t-1
                    input = data[t-1] 
                    model = hospital.models[t-1]
                    temp = model.rnn[0].hidden_state
                    output= model(input)
                    model.rnn[0].hidden_state = temp
                    hloss[i] +=  mael(output, data[t])

    # Calculate and save the average MAE Loss for each hospital
    avg_hloss = np.array([loss.cpu()/val_hos for loss in hloss])
    return avg_hloss



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
            x = x.to(device)
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
            x=x.to(device)
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
