import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from config import *

#Create data objects for the DGN
#https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
#Create data objects for the DGN
#https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
def cast_data(array_of_tensors, subject_type = None, flat_mask = None):
    N_ROI = array_of_tensors[0].shape[0]
    CHANNELS = array_of_tensors[0].shape[2]
    
    dataset = []
    for mat in array_of_tensors: # mat.shape: (35, 35, 4)
            #Allocate numpy arrays 
            edge_index = np.zeros((2, N_ROI * N_ROI))
            edge_attr = np.zeros((N_ROI * N_ROI,CHANNELS)) 
            x = np.zeros((N_ROI, 1))
            x = np.zeros((N_ROI, 1))
            y = np.zeros((1,))
            
            counter = 0
            for i in range(N_ROI):
                for j in range(N_ROI):
                    edge_index[:, counter] = [i, j]
                    edge_attr[counter, :] = mat[i, j]
                    counter += 1

            # Fill node feature matrix (no features every node is 1)
            for i in range(N_ROI):
                x[i,0] = 1
                
            #Get graph labels
            y[0] = None
            
            if flat_mask is not None: 
                edge_index_masked = []
                edge_attr_masked = []
                for i,val in enumerate(flat_mask):
                    if val == 1:
                        edge_index_masked.append(edge_index[:,i])
                        edge_attr_masked.append(edge_attr[i,:])
                edge_index = np.array(edge_index_masked).T
                edge_attr = edge_attr_masked
            
            edge_index = torch.tensor(edge_index, dtype = torch.long)
            edge_attr = torch.tensor(edge_attr, dtype = torch.float)
            x = torch.tensor(x, dtype = torch.float)
            y = torch.tensor(y, dtype = torch.float)
            con_mat = torch.tensor(mat, dtype=torch.float) 
            data = Data(x = x, edge_index=edge_index, edge_attr=edge_attr, con_mat = con_mat,  y=y, label = subject_type)
            dataset.append(data)
    return dataset # graph list


            
def generate_cbt_median(model, train_data):
    """
        Generate optimized CBT for the training set (use post training refinement)
        Args:
            model: trained DGN model
            train_data: list of data objects
    """
    model.eval()
    cbts = []
    train_data = [d.to(device) for d in train_data]
    for data in train_data:
        cbt = model(data)
        cbts.append(np.array(cbt.cpu().detach()))
    final_cbt = torch.tensor(np.median(cbts, axis = 0), dtype = torch.float32).to(device)
    return final_cbt 



def mean_frobenious_distance(generated_cbt, test_data):
    """
        Calculate the mean Frobenious distance between the CBT and test subjects (all views)
        Args:
            generated_cbt: trained DGN model
            test_data: list of data objects
    """
    frobenius_all = []
    for data in test_data:
        views = data.con_mat
        for index in range(views.shape[2]):
            diff = torch.abs(views[:,:,index] - generated_cbt)
            diff = diff*diff
            sum_of_all = diff.sum()
            d = torch.sqrt(sum_of_all)
            frobenius_all.append(d)
    return sum(frobenius_all) / len(frobenius_all)



def generate_subject_biased_cbts(model, train_data):
    """
        Generates all possible CBTs for a given training set.
        Args:
            model: trained DGN model
            train_data: list of data objects
    """
    model.eval()
    cbts = np.zeros((model.model_params["N_ROIs"],model.model_params["N_ROIs"], len(train_data)))
    train_data = [d.to(device) for d in train_data]
    for i, data in enumerate(train_data):
        cbt = model(data)
        cbts[:,:,i] = np.array(cbt.cpu().detach())

    return cbts



#Clears the given directory
def clear_dir(dir_name):
    for file in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, file))
        
        
        
def plotLosses(loss_table_list):
    '''
    This function plots every model's every fold's loss performance and saves them with their particular information written with their names.
    '''
    for i in range(n_folds):
        cur_loss_table = loss_table_list[i]
        for k in range(number_of_samples):
            if isFederated:
                fig1, ax1 = plt.subplots()
                loss_lst = cur_loss_table['combining_local_loss_global_data_'+str(k)]
                ax1.plot((np.arange(len(loss_lst)) + 1) * numEpoch, loss_lst)
                ax1.set(xlabel='epochs', ylabel='rep loss', title='{}th Fold {}th Client Combining Local Loss Global Data {}'.format(i,k, "%.4f" %min(loss_lst)))
                ax1.grid()
                fig1.savefig('{}fold{}_{}th_client_combining_local_loss_global_data.png'.format(Path_output, i, k))
                plt.show()

            fig2, ax2 = plt.subplots()
            loss_lst = cur_loss_table['local_loss_global_data_'+str(k)]
            ax2.plot((np.arange(len(loss_lst)) + 1) * numEpoch, loss_lst)
            ax2.set(xlabel='epochs', ylabel='rep loss', title='{}th Fold {}th Client Local Loss Global Data {}'.format(i,k, "%.4f" %min(loss_lst)))
            ax2.grid()
            fig2.savefig('{}fold{}_{}th_client_local_loss_global_data.png'.format(Path_output, i, k))
            plt.show()


            
def show_image(img, i, k):
    img = np.repeat(np.repeat(img, 10, axis=1), 10, axis=0)
    plt.imshow(img)
    plt.title("Fold " + str(i) + " Client " + str(k))
    plt.axis('off')
    if not os.path.exists('output/' + Dataset_name):
        os.mkdir('output/' + Dataset_name)
    if not os.path.exists('output/' + Dataset_name + '/' + Setup_name):
        os.mkdir('output/' + Dataset_name + '/' + Setup_name)
    plt.savefig('output/{}/{}/fold{}_cli_{}_{}_DGN_cbt.jpg'.format(Dataset_name, Setup_name, i, i, k, Setup_name), bbox_inches='tight')
    
    
    
#Antivectorize given vector (this gives a symmetric adjacency matrix)
def antiVectorize(vec, m):
    
    #Old Code
    M = np.zeros((m,m))
    M[np.triu_indices(m)] = vec
    M[np.tril_indices(m)] = vec
    M[np.diag_indices(m)] = 0
    return M

def simulate_dataset(N_Subjects, N_Nodes, N_views):
    """
        Creates random dataset
        Args:
            N_Subjects: number of subjects
            N_Nodes: number of region of interests
            N_views: number of views
        Return:
            dataset: random dataset with shape [N_Subjects, N_Nodes, N_Nodes, N_views]
    """
 
    features =  np.triu_indices(N_Nodes)[0].shape[0]
    views = []
    for _ in range(N_views):
        view = np.random.uniform(0.1,2, (N_Subjects, features))
        
        view = np.array([antiVectorize(v, N_Nodes) for v in view])
        views.append(view)
    return np.stack(views, axis = 3)
