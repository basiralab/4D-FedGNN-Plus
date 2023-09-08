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
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
import networkx as nx
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
    
    

def node_features_from_adj_matrix(adj_matrix,device):

      if device.type=='cpu':
          # Create a NetworkX graph from the adjacency matrix
          G = nx.from_numpy_array(adj_matrix.detach().numpy())
      elif device.type=='cuda':
          # Create a NetworkX graph from the adjacency matrix
          G = nx.from_numpy_array(adj_matrix.detach().cpu().numpy())

      # Compute the weighted degree (strength) for each node
      strength = dict(G.degree(weight='weight'))

      # Compute the degree for each node
      degree = dict(G.degree())

      # Compute the clustering coefficient for each node
      clustering = nx.clustering(G, weight='weight')

      # Compute the closeness centrality for each node
      closeness_centrality = nx.closeness_centrality(G)

      # Compute the PageRank for each node
      pagerank = nx.pagerank(G, weight='weight')

      # Let's convert these features into numpy arrays so we can stack them together
      strength_array = np.array(list(strength.values()))
      degree_array = np.array(list(degree.values()))
      clustering_array = np.array(list(clustering.values()))
      closeness_centrality_array = np.array(list(closeness_centrality.values()))
      pagerank_array = np.array(list(pagerank.values()))

      # Now we can stack these features together to get a node feature matrix
      x = torch.Tensor(np.vstack([strength_array, degree_array, clustering_array, closeness_centrality_array, pagerank_array]).T)
      return x
      
      
def adj_matrix_to_pytorch_geometric_data(adj_matrix,device):

    # calculate edge_index and edge_weights
    edge_indices, edge_weights = dense_to_sparse(adj_matrix)

    #edge attributes
    edge_attr = torch.cat([edge_indices.T,edge_weights.view(len(edge_weights),1)],1)

    # calculate the node features
    x = node_features_from_adj_matrix(adj_matrix,device)

    data = Data(x=x.to(device),edge_index=edge_indices.to(device),edge_weights=edge_weights.to(device),adj_matrix=adj_matrix.to(device),edge_attr=edge_attr.to(device))

    return data


def create_edge_index_attribute_new(adj_matrix):
    """
    Given an adjacency matrix, this function creates the edge index and edge attribute matrix
    suitable to graph representation in PyTorch Geometric.
    """

    rows, cols = adj_matrix.shape[0], adj_matrix.shape[1]
    edge_index = [[],[]]
    edge_attr = []

    for i in range(rows):
        for j in range(cols):

              if adj_matrix[i,j] > 0:

                    edge_index[0].append(i)
                    edge_index[1].append(j)
                    edge_attr.append(adj_matrix[i,j])

    return torch.tensor(edge_index), torch.Tensor(edge_attr), rows, cols
    
    

def get_adjacency_matrix(num_nodes, edge_indices, edge_weights):
    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # Fill in the adjacency matrix
    for ((node1, node2), weight) in zip(edge_indices, edge_weights):
        adjacency_matrix[node1, node2] = weight

    return adjacency_matrix
    
   
def train_gnns_final(args, dataset,seed=10,ratio=4/8,verbose=False,train_validate_verbose=True,train_validate_verbosity_epochs=1):
        """
            Arguments:
            args: arguments
            dataset: the whole dataset (train and test set)
            table: [num_hospitals, num_timepoints], holds timepoint-wise availability of hospitals

        This function performs training and testing reporting Mean Absolute Error (MAE) of the testing brain graphs.
        """

        # Create the results folders
        print('Train NEW')
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.save_path+f'{args.save_name}', exist_ok=True)
        os.makedirs(args.save_path+f'{args.save_name}/real_and_predicted_graphs', exist_ok=True)
        os.makedirs(args.save_path+f'{args.save_name}/train_losses', exist_ok=True)
        os.makedirs(args.save_path+f'{args.save_name}/train_losses/mae_losses', exist_ok=True)
        os.makedirs(args.save_path+f'{args.save_name}/train_losses/tp_losses', exist_ok=True)
        os.makedirs(args.save_path+f'{args.save_name}/train_losses/total_losses', exist_ok=True)
        os.makedirs(args.save_path+f'{args.save_name}/test_mae_losses', exist_ok=True)
        os.makedirs(args.save_path+f'{args.save_name}/trained_models', exist_ok=True)


        # Create the results folders
        manualSeed = seed

        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        # show the fixed seed
        print(f'Fixed seed:{seed}')
        print()

        # create the table that shows the data availability by  timepoint
        table = np.zeros((args.num_folds - 1, args.num_timepoints))
        table = random_table(args, ratio)
        print(f'Table:')
        print(table)
        print(f'Ratio:{ratio}')
        print()

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('running on GPU')
            # if you are using GPU
            torch.cuda.manual_seed(manualSeed)
            torch.cuda.manual_seed_all(manualSeed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # !!! necessary for the below line
            torch.use_deterministic_algorithms(False)
            print('TRAIN deterministic algorithms')

        else:
            device = torch.device("cpu")
            print('running on CPU')

        # change the save path
        args.save_path = args.save_path+f'{args.save_name}/'

        # Choosing the right get_order function
        if args.mode == 'weighted_weight_exchange':
            get_order =  get_order_weighted
        else:
            get_order =  get_order_gnns


        # Getting our fold dict
        fold_dict,X = mf.create_fold_dict_new(dataset,num_hospitals=4,num_folds=5)

        # Perform the 5-fold Cross-Validation
        num_hospitals = args.num_folds - 1
        for f in range(args.num_folds):

            # fix the seeds
            np.random.seed(manualSeed)
            random.seed(manualSeed)
            torch.manual_seed(manualSeed)

            tic0 = timeit.default_timer()

            print(
                f'------------------------------------Fold [{f + 1}/{args.num_folds}]-----------------------------------------')

                        # Create hospitals
            hospitals = []
            for i in range(num_hospitals):
                  hospitals.append(Hospital_meta(args,device))
                  print('GAT')


            # Train data for each hospital
            train_data_list = []

            # Current test data
            test_data = X[-1][f].to(device)

            for i in range(num_hospitals):

                # Append the fold related to Hospital i - fold f. Note, here we append the real data, not the indices
                train_data_list.append(X[i][fold_dict[f'Hospital_{i}'][f'fold_{f}'][0]].to(device))


            # Start measuring the epochs time
            epochs_start = time.time()

            # Initiate Training
            for epoch in range(args.num_epochs):

                  epoch +=1

                  # order the hospitals based on the data availability
                  ordered_hospitals = get_order_gnns(table)

                  for h_i,hospital in enumerate(hospitals):

                    # get the train data for the hospital
                    train_data = train_data_list[h_i]

                    # get the table for th current hospital
                    table_hospital = table[h_i]

                    #if h_i ==3:
                     # return args,hospital,train_data,table_hospital

                    # Train the current hospital at the current timepoint for 1 epoch
                    mae_loss,hospital = train_one_epoch_gnns(args,hospital,train_data,table_hospital)

                    # Updating the hospital
                    hospitals[h_i] = hospital

                    if verbose:
                        print(f'Epoch:{epoch} Hospital:{h_i}, Train MAE Loss:{mae_loss}')


                  # Perform validation during training
                  if train_validate_verbose and (epoch%train_validate_verbosity_epochs==0 or epoch==1):

                      val_loss,val_mean_loss = validate_during_training_gnns(args, hospitals, test_data)
                      print(f"Epoch:{epoch},val_loss:")
                      print(f'Total MAE Loss')
                      print(val_loss)
                      print()
                      print(f'Average MAE Loss:')
                      print(val_mean_loss)
                      print()

                      for h_i,l in enumerate(val_mean_loss):

                          hospitals[h_i].scheduler.step(l)


                  if epoch != args.num_epochs - 1 or epoch != 0:
                      if epoch % args.C == 0 and args.mode != "4D-GNN":
                          print('Central Aggregation')
                          hospitals = update_main_by_average_gnns(hospitals)
                      elif epoch % args.D == 0 and args.mode == "4D-FED-GNN+":
                          hospitals = exchange_models(hospitals, t)
                      elif epoch % args.D == 0 and args.mode == "4D-FED-GNN++":
                          print('4D-FED-GNN++')
                          hospitals = exchange_models_based_on_order_gnns(hospitals, ordered_hospitals)
                      elif epoch % args.D == 0 and args.mode == "weighted_weight_exchange":
                          print('weighted_weight_exchange')
                          hospitals = exchange_models_weights_pairs_extreme(hospitals, t, ordered_hospitals)



            epochs_end = time.time()-epochs_start
            print()
            print(f'epochs finished with time:{epochs_end}')
            print()
            validate_gnns(args, hospitals, test_data, f)
            tic1 = timeit.default_timer()
            timer(tic0,tic1)


def exchange_models_based_on_order_gnns(hospitals, order):
    """
        This function exchanges GNN-layer weights of paired hospitals at timepoint t with each other
    """
    pre_model = None
    for i, h_i in enumerate(order):
        next_model = copy.deepcopy(hospitals[h_i].model.state_dict())

        if not pre_model is None:
            hospitals[h_i].model.load_state_dict(pre_model)

        pre_model = copy.deepcopy(next_model)

        if i == 0:
            hospitals[h_i].model.load_state_dict(copy.deepcopy(hospitals[order[-1]].model.state_dict()))

    return hospitals
    

def update_main_by_average_gnns(hospitals):
    """
        This function takes the GNN-layer weights of the GNN at timepoint t and computes the global model by averaging,
        then broadcats the weights to the hospitals (updates each GNN with the global model)
    """
    target_state_dict = copy.deepcopy(hospitals[0].model.state_dict())
    mux = 1 / len(hospitals)


    model_state_dict_list = [copy.deepcopy(hospitals[i].model.state_dict()) for i in range(1, len(hospitals))]


    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data = target_state_dict[key].data.clone() * mux
            for model_state_dict in model_state_dict_list:
                target_state_dict[key].data += mux * model_state_dict[key].data.clone()

    for i in range(len(hospitals)):
        hospitals[i].model.load_state_dict(target_state_dict)

    return hospitals
    
    
def validate_during_training_gnns(args, hospitals, test_data):
    """
    This function calculates the average MAE of predicted brain graphs during validation.
    """
    mael = torch.nn.L1Loss()

    hloss=np.array([np.zeros(args.num_timepoints-1) for i in range(args.num_hospitals)])
    for h_i, hospital in enumerate(hospitals):
           hospital.model.eval()

           with torch.no_grad():


                for data in test_data:

                    input = data[0]
                    mae_loss_hospital = np.zeros(args.num_timepoints-1)

                    for t in range(args.num_timepoints-1):

                        pred = hospital.model(input)
                        hloss[h_i,t] +=  mael(pred, data[t+1])
                        input = pred


    # Calculate and save the average MAE Loss for each hospital
    avg_hloss = hloss/len(test_data)
    avg_hloss_mean = np.mean(avg_hloss,axis=1)
    return avg_hloss,avg_hloss_mean
    

def validate_gnns(args, hospitals, test_data,f,verbose=False):
    """
    This function calculates the average MAE of predicted brain graphs during validation.
    """
    mael = torch.nn.L1Loss()

    hloss=np.array([np.zeros(args.num_timepoints-1) for i in range(args.num_hospitals)])
    for h_i, hospital in enumerate(hospitals):
           hospital.model.eval()

           with torch.no_grad():


                for subject_index,data in enumerate(test_data):

                    input = data[0]
                    mae_loss_hospital = np.zeros(args.num_timepoints-1)

                    for t in range(args.num_timepoints-1):

                        pred = hospital.model(input)
                        hloss[h_i,t] +=  mael(pred, data[t+1])
                        input = pred

                        #plot and save the brain graph of patient(sample) i through all the timepoints
                        plot_matrix(data[t+1].cpu().detach().numpy(),f'Real Graph, Hospital:{h_i}, Subject:{subject_index}, Timepoint:{t+1}',
                                  args.save_path+'real_and_predicted_graphs/'+f'hospital_{h_i}_subject_{subject_index}_timepoint_{t+1}_fold_{f}_real_graph',verbose)
                        plot_matrix(pred.cpu().detach().numpy(),f'Predicted Graph, Hospital:{h_i}, Subject:{subject_index}, Timepoint:{t+1}',
                                  args.save_path+'real_and_predicted_graphs/'+f'hospital_{h_i}_subject_{subject_index}_timepoint_{t+1}_fold_{f}_predicted_graph',verbose)



    # Calculate and save the average MAE Loss for each hospital
    avg_hloss = hloss/len(test_data)
    for h_i,loss in enumerate(avg_hloss):

        # Save the MAE Loss
        np.save(args.save_path+f"test_mae_losses/mae_test_loss_hospital_{h_i}_fold_{f}", loss)
    # Save the loss

    print(avg_hloss)
    return avg_hloss
    

class Hospital_gnns():
    def __init__(self, args,device):
        """
        Hospital object contains a GNN and an optimizer for each timepoint

        Hospital object can update GNN-layer wise weights of its GNNs
        """

        self.model = RGCN(in_channels=3,hidden_size=32,num_nodes=35,window=1,dropout=0.4,device=device).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=args.lr)
        

def train_one_epoch_gnns(args,hospital,train_data,table_hospital):
      # Set the model in training mode
      hospital.model.train()

      num_timepoints = train_data.shape[1]

      #loss types definition
      MAE_loss = torch.nn.L1Loss()

      # this is our loss for all the data

      mae_loss_overall = []

      # loop through the data batches
      for data_id,data in enumerate(train_data):

            # zero the gradients
            hospital.optimizer.zero_grad()

            num_preds=0 # the number of times that we are able to predict
            mae_loss = 0
            t=0

            # loop through the time dependent adj matrices in the batches
            while t < num_timepoints-1:
                #print(data_id,t)

                # check if the next timepoint is available
                if table_hospital[t+1]==1:


                    pred = hospital.model(data[t])
                    real = data[t+1]

                    mae_loss += MAE_loss(pred,real)
                    #print(f'MAE LOSS FROM CURRENT PREDICTION:{MAE_loss(pred,real) }')
                    num_preds+=1
                    t+=1

                # if the next timepoint is not available
                elif table_hospital[t+1]==0:

                    pred = hospital.model(data[t])
                    #return pred


                    # find the next closes available 1 to use it for a label. We use the pred as input until then
                    reached = False
                    for t in range(t+1,num_timepoints-1):
                        #print(f'Search t:{t}')


                        # using the previous prediction as input
                        pred = hospital.model(pred)

                        # if the next timepoint has data available we break
                        if table_hospital[t+1]==1:

                                real = data[t+1]
                                mae_loss += MAE_loss(pred,real)
                                #print(f'MAE LOSS FROM CURRENT PREDICTION:{MAE_loss(pred,real)}')
                                num_preds+=1
                                t+=1
                                reached = True
                                break

                    # if there are no more 1s in our table, we break from the loop and use whatever losses we accumulated in this loop
                    if not reached:
                      break

                #print()
                #print(f'the new t is:{t}')
                #print()

            #print(mae_loss)
            # Calculate the total MAE Loss for the current batch

            if num_preds == 0:
              return 100,hospital

            mae_loss=mae_loss/num_preds
            # Append to the total MAE Los
            mae_loss_overall.append(mae_loss.item())
            #print(f'Num Predictions:{num_preds}')

            # Update the weights of the neural network
            mae_loss.backward()
            hospital.optimizer.step()



      mae_loss_overall = np.mean(np.array(mae_loss_overall))
      return mae_loss_overall,hospital
      
def get_order_gnns(table):

    hospital_distances=[]
    hospital_number_of_points=[]
    for row in table:

        hospital_number_of_points.append(int(row.sum()))
        ones_indexes = np.where(row==1)[0]
        distances = np.diff(ones_indexes)-1
        distance = distances.sum()
        hospital_distances.append(distance)


    hospital_number_of_points = np.array(hospital_number_of_points)
    hospital_distances = np.array(hospital_distances)
    order = np.lexsort((-hospital_distances,hospital_number_of_points))
    order = np.flip(order)



    return order


    
######################################################## 4D-FED-GNN++ ######################################################################
    
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
