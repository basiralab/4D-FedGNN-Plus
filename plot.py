import argparse
import os
from PIL import Image
import io
import os.path as osp
import numpy as np
import math
import itertools
import pandas as pd
import seaborn as sns
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, LeakyReLU
from torch.autograd import Variable

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import NNConv, BatchNorm, EdgePooling, TopKPooling, global_add_pool

import matplotlib.pyplot as plt


def plot(losses, title, loss,save_path,verbose):
    fig = plt.figure()
    plt.plot(losses)
    plt.xlabel("# epoch")
    plt.ylabel(loss)
    plt.title(title)
    plt.savefig(save_path+'.png')
    if verbose:
        plt.show()
    plt.close()


def plot_matrix(out, strategy,save_path,verbose):
    fig = plt.figure()
    plt.pcolor(abs(out))
    plt.colorbar()
    plt.imshow(out)
    title = strategy
    plt.title(title)
    if verbose:
        plt.show()
    plt.savefig(save_path + '.png')
    plt.close()        
                
                
def plot_test_loss(selected_folders,base_dir='/Users/pavelbozmarov/Desktop/Python_Projects/Imperial/Dissertation/Code/4D-FedGNN-Plus_mine/Real_Overall/4_8/4D-FED-GNN++/',percentage=50):

    # We need to have a color for each method. 
    colors = ['#FF69B4', '#00FFFF', '#00FF00', '#FF0000', '#FFFF00', '#800080', '#000080', '#008000', '#FFA500', '#800000', '#808000', '#008080', '#0000FF', '#A52A2A', '#FA8072', '#7B68EE', '#8B4513', '#2E8B57', '#ADFF2F', '#D2691E']
    
    def compute_results(dir_path):
            '''
            Creates a results dictionary suitable for bar graph plots for a given directory.
            The keys of the dictionary are the different timepoints and for each timepoint we have
            an array as value and the item corresponding to index i of this array is the mae test error
            for hospital i for this timepoint.
            '''
            
            # Initialize an empty dictionary to store the data
            data = {}

            # For each file in the directory
            for filename in os.listdir(dir_path):
                print()
                # If the file is a .npy file
                if filename.endswith(".npy"):
                    # Extract the hospital and fold number from the filename
                    hospital_num = int(filename.split("_")[4])
                    fold_num = int(filename.split("_")[6].split(".")[0])

                    # Load the .npy file
                    file_path = os.path.join(dir_path, filename)
                    mae_test_loss = np.load(file_path)

                    # Add the data to the dictionary, creating a new dictionary for each fold if necessary
                    fold_key = f"fold_{fold_num}"
                    if fold_key not in data:
                        data[fold_key] = {}
                    hospital_key = f"hospital_{hospital_num}"
                    data[fold_key][hospital_key] = mae_test_loss

                
                # Initialize a dictionary to store the average results
            average_results = {}

            # For each fold in the data
            for fold_key in data:
                # For each hospital in the fold
                for hospital_key in data[fold_key]:
                    # If the hospital is not yet in the average_results dictionary, add it with the current mae_test_loss
                    if hospital_key not in average_results:
                        average_results[hospital_key] = [data[fold_key][hospital_key]]
                    # If the hospital is already in the average_results dictionary, append the current mae_test_loss
                    else:
                        average_results[hospital_key].append(data[fold_key][hospital_key])
        
            # Now compute the average for each hospital
            for hospital_key in average_results:
                average_results[hospital_key] = np.mean(average_results[hospital_key], axis=0)

            # sort hospitals by their numbers
            sorted_hospitals = sorted(average_results.items(), key=lambda x: int(x[0].split('_')[1]))

            # Get the length of the array for any one hospital
            num_timepoints = len(average_results['hospital_0'])

            # Initialize new dictionary
            new_result = {}

            # Dynamically create the timepoints
            for i in range(num_timepoints):
                new_result['timepoint_{}'.format(i+1)] = []

            # Populate the new dictionary
            for hospital, values in sorted_hospitals:
                for i in range(num_timepoints):
                    new_result['timepoint_{}'.format(i+1)].append(values[i])
            
            return new_result
    
    folder_data = {}
    for folder_name in selected_folders:
            dir_path = os.path.join(base_dir, folder_name, 'overall_test_mae_losses')
            result = compute_results(dir_path)
            folder_data[folder_name]=result
   
    
    # convert to appropriate dictionary form
    new_data = {}
    first_key = list(folder_data.keys())[0]
    for sub_key in folder_data[first_key]:
        new_data[sub_key] = {}
        for key, value in folder_data.items():
            new_data[sub_key][key] = value[sub_key]


    # convert to appropriate dataframe form in order to plot with seaborn
    
    # Create an empty DataFrame
    df = pd.DataFrame()

    # Loop through the dictionary to fill the DataFrame
    for timepoint, datasets in new_data.items():
        for key_folder, losses in datasets.items():
            temp_df = pd.DataFrame()
            temp_df['timepoint'] = [timepoint] * len(losses)
            temp_df['method'] = [key_folder] * len(losses)
            temp_df['hospital'] = ['Hospital ' + str(i) for i in range(len(losses))]
            temp_df['MAE Loss'] = losses
            temp_df['mean MAE Loss'] = np.mean(losses)  # Add mean loss calculation
            df = pd.concat([df, temp_df])


    ###################################### PLOT ######################################
    figs = []
    statistics_dfs = []  # Initialize an empty list to store the DataFrames with the statistics
    for i in range(len(set(df['timepoint']))):
        curr_df = df[df['timepoint'] == f'timepoint_{i+1}']
        min_mae = curr_df['MAE Loss'].min()*0.99
        fig = plt.figure(figsize=(15, 15))  
        ax = fig.add_subplot(111)
        sns.barplot(x="hospital", y="MAE Loss", hue="method", data=curr_df, palette=colors[:len(set(df['method']))], ax=ax, width=0.5,alpha=1,zorder=3)

        # get unique methods and hospitals
        methods = curr_df['method'].unique()
        hospitals = curr_df['hospital'].unique()

        # width of each bar
        bar_width = 0.5 / len(methods)
        

        mean_values = curr_df.groupby('method')['MAE Loss'].mean()
        std_values = curr_df.groupby('method')['MAE Loss'].std()
        
        # Create a DataFrame to print the mean and std values in a nice table format
        statistics_df = pd.DataFrame({'Mean MAE Loss': mean_values, 'STD MAE Loss': std_values})

        # Add the statistics_df to the list of statistics_dfs
        statistics_dfs.append(statistics_df)
        
        ax.set_title(f'Real Data, Missing data: {percentage}%, Timepoint {i+1}', fontweight='bold', fontsize=16)
        ax.set_ylabel('MAE', fontweight='bold',fontsize=16)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold', fontsize=16)  # Increase the size of the x-tick labels

        # Change the plot background color
        ax.set_facecolor('lightgrey')  # Change 'lightgrey' to your desired color

        # Change the color of the horizontal gridlines
        ax.yaxis.grid(color='white')

        # Position the legend below the plot without a box
            
                # Position the legend below the plot without a box
        legend1 = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, prop={'size': 18})
        ax.add_artist(legend1)
        import matplotlib.lines as mlines
        import matplotlib as mpl
        import matplotlib.patches as mpatches

        # Make sure the full figure (including legend) is saved, not just the axes bbox
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)  # Adjust this as needed to make room for the legend

        # Add black borders to the legend color boxes
        for legend_handle in legend1.legend_handles:
            legend_handle.set_edgecolor('black')
            legend_handle.set_linewidth(0.5)

            # Add borders to the bars
        for container in ax.containers:

            if not isinstance(container, mpl.container.ErrorbarContainer):
                for rect in container:
                    rect.set_zorder(10)
                    rect.set_linewidth(0.5)
                    rect.set_edgecolor('black')

        # Remove label from x-axis
        ax.set_xlabel('')

        # Make the x-tick labels bold
        ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')

        # Adjust the y-axis limits to remove the slight tick above the bars
        ax.set_ylim(bottom=min_mae)

        #plt.show()

        # Save the current plot as a PIL image and append to the list
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        figs.append(Image.open(buf))
        plt.close()  # Close the current plot before creating a new one
    
    # sort the df my the mae loss
    statistics_dfs = [df.sort_values(by="Mean MAE Loss", ascending=True) for df in statistics_dfs]
    return figs, statistics_dfs