import numpy as np
from sklearn.model_selection import KFold
from collections import defaultdict
from itertools import combinations
import torch
import copy
import scipy


def create_fold_dict(dataset,num_hospitals=4,num_folds=5):
    """
    This function creates a dictionary of the folds for each hospital
    :param dataset: The dataset to be split into folds
    :param num_folds: The number of folds to split the dataset into
    :param num_hospitals: The number of hospitals in the dataset
    :return: A dictionary of the folds for each hospital
    """
    # Split dataset into 4 for the hospitals and calculate the fold indices for each hospital using a dictionary
    X = np.array_split(dataset, num_hospitals)
    kf = KFold(n_splits=num_folds, shuffle=False)
    fold_dict = defaultdict(lambda: defaultdict(dict))
    print()
    print('Creating the fold dictionary...')
    print()
    for j,data in enumerate(X):

          print(f'Data for hospital {j}')

          for i,item in enumerate(list(kf.split(data))):

                print(f'fold:{i}')

                fold_dict[f'Hospital_{j}'][f'fold_{i}'] = item


    return fold_dict,X


def create_fold_dict_new(dataset,num_hospitals=4,num_folds=5):
    """
    This function creates a dictionary of the folds for each hospital and one more fold for global testing
    The global testing fold itself is split into num_folds
    :param dataset: The dataset to be split into folds
    :param num_folds: The number of folds to split the dataset into
    :param num_hospitals: The number of hospitals in the dataset
    :return: A dictionary of the folds for each hospital and list of all the folds
    """

    # Split dataset into num_folds for the hospitals and calculate the fold indices for each hospital using a dictionary
    dataset = copy.deepcopy(dataset)
    # convert the dataset to numpy to avoid mistakes for the further code
    dataset = dataset.cpu().numpy()
    
    np.random.shuffle(dataset)

    # Split the data into num_hospitals+1 portions
    X = np.array_split(dataset, num_hospitals+1)

    # Split the last fold of X into num_folds -> this will be our global test set
    X[-1] = np.array_split(X[-1],num_folds)

    # prepare the Cross-Validation
    kf = KFold(n_splits=num_folds, shuffle=True)
    fold_dict = defaultdict(lambda: defaultdict(dict))
    print()
    print('Creating the fold dictionary...')
    print()
    for j,data in enumerate(X):

            if j == num_hospitals:

                    continue 

            print(f'Data for hospital {j}')

            for i,item in enumerate(list(kf.split(data))):

                    print(f'fold:{i}')

                    fold_dict[f'Hospital_{j}'][f'fold_{i}'] = item
     
     # Convert X to torch
    for i in range(len(X)):
          if i == len(X)-1:
                for j in range(len(X[-1])):

                    X[-1][j] = torch.from_numpy(X[-1][j])
                    X[-1][j] = X[-1][j].type(torch.FloatTensor)
                        
          else:
                X[i] = torch.from_numpy(X[i])
                X[i] = X[i].type(torch.FloatTensor)

     
    return fold_dict,X

def create_train_fold_dict(X, fold_dict, num_folds=5, num_hospitals=4):
    """
    This function creates a dictionary of the training data for each fold and hospital.

    Parameters:
    X: list of numpy arrays - The split data for each hospital.
    fold_dict: dict - The fold dictionary created by the create_fold_dict function.
    num_folds: int - The number of folds.
    num_hospitals: int - The number of hospitals.

    Returns:
    train_dict: defaultdict - A dictionary where each hospital and fold indexes the corresponding training data.
    """
    train_dict = defaultdict(lambda: defaultdict(dict))

    for f in range(num_folds):
        for i in range(num_hospitals):
            train_dict[f'Hospital_{i}'][f'fold_{f}'] = X[i][fold_dict[f'Hospital_{i}'][f'fold_{f}'][0]]

    return train_dict


def euclidean_distance(data1, data2):
    # Calculates the euclidean distance between 2 arrays/vectors/data points
    return np.linalg.norm(data1-data2)

def spectral_distance(A1, A2):

    A1 = np.array(copy.deepcopy(A1))
    A2 = np.array(copy.deepcopy(A2))

    # Degree matrices
    D1 = np.diag(np.sum(A1, axis=1))
    D2 = np.diag(np.sum(A2, axis=1))

    # Laplacian matrices
    L1 = D1 - A1
    L2 = D2 - A2

    # Eigenvalues
    eigenvalues1 = scipy.linalg.eigvalsh(L1)
    eigenvalues2 = scipy.linalg.eigvalsh(L2)

    # Euclidean distance between eigenvalue sequences
    distance = np.linalg.norm(eigenvalues1 - eigenvalues2)

    return distance

def calculate_dissimilarities(data,mode='euclidean'):
    """
    Calculate the average pairwise Euclidean distance between the given data points.
    
    :param data: A list or array of data points.
    
    :return: The average pairwise Euclidean distance, rounded to 2 decimal places.
    """
    measure = None
    if mode=='euclidean':
        measure = euclidean_distance
    elif mode =='spectral':
        measure = spectral_distance

    # Compute the pairwise distances
    pairwise_distances = [measure(pair[0], pair[1]) for pair in combinations(data, 2)]
    
    # Return the average distance, rounded to 2 decimal places
    return round(np.mean(pairwise_distances), 2)  


def create_dissimilarity_dict(train_dict,mode,num_folds=5,num_hospitals=4,num_timepoints=3):

    '''Creates dissimilarity dictionary.
       Input: train_dict - dictionary that contains the train data by folds and hospitals
    '''
    # we will create a dictionary of the train fold data
    dissimilarities = defaultdict(lambda: defaultdict(dict))

    for f in range(num_folds):
        #print(f'fold:{f}')

        for i in range(num_hospitals):
            #print(f'Hospital:{i}')

            for t in range(num_timepoints):
                dissimilarities[f'Hospital_{i}'][f'fold_{f}'][f'timepoint_{t}'] = calculate_dissimilarities(train_dict[f'Hospital_{i}'][f'fold_{f}'][:,t,:,:],mode)

    return dissimilarities


def create_dissimilarity_table(dissimilarities, fold):
    """
    Creates a table of dissimilarities for each hospital at each timepoint, for a given fold.

    :param dissimilarities(dict): A dictionary containing the dissimilarity data. The keys should be in the form 
                            'Hospital_' and the values should be dictionaries with fold and timepoint keys.
    :param fold(str): The fold to use when extracting dissimilarity data. Ex: 'fold_0'

    :return(arr): An array of arrays, where each array contains the dissimilarities for a hospital at each timepoint. 
             Each value is rounded to 2 decimal places.
    """

    # Get a sorted list of all hospitals in the dissimilarities dictionary.
    hospitals = sorted([k for k in dissimilarities.keys() if f"Hospital_" in k])
    
    # Get a sorted list of all timepoints in the dissimilarities dictionary for the first hospital.
    # We assume that all hospitals have the same timepoints.
    timepoints = sorted([k for k in dissimilarities[hospitals[0]][fold].keys() if "timepoint_" in k])
    
    # Initialize the dissimilarity table as an empty list.
    dissimilarity_table = []

    # Iterate over all hospitals.
    for hospital in hospitals:
        # Initialize a list to hold the dissimilarities for this hospital.
        hospital_dissimilarities = []
        
        # Iterate over all timepoints.
        for timepoint in timepoints:
            # Append the dissimilarity for this hospital and timepoint.
            hospital_dissimilarities.append(dissimilarities[hospital][fold][timepoint])
        
        # Add the list of dissimilarities for this hospital to the dissimilarity table.
        dissimilarity_table.append(hospital_dissimilarities)
    
    return np.array(dissimilarity_table)


def dissimilarity_order(dissimilarity_table):
    """
    Returns the indices of rows in a dissimilarity table ordered by their sums.

    :param dissimilarity_table: A 2D NumPy array of dissimilarities.
    :return: Indices of rows in descending order of their sums.
    """
    # Compute the sum of dissimilarities for each row.
    sums = np.sum(dissimilarity_table, axis=1)

    # Get the indices that would sort the sums in ascending order.
    order = np.argsort(sums)

    # Reverse the order to get indices in descending order.
    order = np.flip(order)

    return order


def get_total_order(diss_order, time_order):
    """
    Calculates a total order based on dissimilarity and time orders.
    
    :param diss_order: A list/array of indices representing dissimilarity order.
    :param time_order: A list/array of indices representing time order.
    :return: Indices representing the total order.
    """
    # Initialize ranking starting from 1 to the number of elements in orders
    ranks = np.flip(np.arange(1, len(time_order) + 1))

    # Initialize a dictionary to store the sum of ranks for each index
    rank_sums = {i: 0 for i in range(len(time_order))}

    # Sum the ranks for each index based on both dissimilarity and time orders
    for i, (h1, h2) in enumerate(zip(diss_order, time_order)):
        rank_sums[h1] += ranks[i]
        rank_sums[h2] += ranks[i]

    # Convert the rank sums to a numpy array
    order = np.array(list(rank_sums.values()))

    # Sort the indices based on the sums in ascending order
    order = np.argsort(order)

    # Reverse the order to get the indices in descending order
    order = np.flip(order)

    return order


def compare_models(model1, model2):
    """
    Compares two PyTorch models based on their parameters.

    Args:
        model1: The first model.
        model2: The second model.

    Returns:
        A boolean value. True if models are equal, otherwise False.
    """
    # Extract model parameters
    model1_params = list(model1.parameters())
    model2_params = list(model2.parameters())

    if len(model1_params) != len(model2_params):
        return False

    for p1, p2 in zip(model1_params, model2_params):
        if not torch.equal(p1, p2):
            return False

    return True


