# Import libraries
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import yaml
with open("/mnt/c/Users/Nico-/Documents/Python_Projects/MaskFormer/output_vitrolife_21_51_08MAR2022/vitrolife_config_initial.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


# Define a function to compute the moving average of an input array or list
def mov_avg_array(inp_array, mov_of_last_n_elements=4, output_last_n_elements=1):                       # Define a function to compute the moving average of an array or a list
    assert output_last_n_elements <= mov_of_last_n_elements, "The moving average can't be outputted for more values than it is being calculated for"
    if mov_of_last_n_elements > len(inp_array): mov_of_last_n_elements = len(inp_array)                 # If the list/array isn't as long as the wanted moving-average value, the n is lowered
    used_array_part = inp_array[-mov_of_last_n_elements:]                                               # Extract the last mov_of_last_n_elements from the list to compute the moving average for
    used_array_cumsum = np.cumsum(used_array_part)                                                      # Compute the cumulated sum for the used array part
    used_array_mov_avg = np.divide(used_array_cumsum, np.arange(1,1+mov_of_last_n_elements))            # Compute the moving average of the used array part
    return used_array_mov_avg[-output_last_n_elements:]                                                 # Output the last output_last_n_elements of the moving average array 


# Define a function to load the metrics.json in each output directory
def load_metrics(config):
    metrics_list = [os.path.join(config["OUTPUT_DIR"], x) for x                                         # Iterate through the files in the output dir of the config file ...
        in os.listdir(config["OUTPUT_DIR"]) if x.endswith(".json") and "metrics" in x.lower()]          # ... and extract the metrics.json filename
    assert len(metrics_list) == 1, "Only one metrics.json file in each output directory is allowed"     # Only one metrics.json file in each output dir
    metrics_file = metrics_list[0]                                                                      # Extract the metrics.json file from the list
    metrics_df = pd.read_json(os.path.join(metrics_file), orient="records", lines=True)                 # Read the metrics.json as a pandas dataframe
    metrics_df.sort_values("iteration")                                                                 # Sort the dataframe by the iteration count
    metrics_df_training = metrics_df[~metrics_df["total_loss"].isna()].dropna(axis=1, how="all")        # The first rows are from the training with all iterations and loss computations
    metrics_dict_training = metrics_df_training.to_dict(orient="list")                                  # Convert the dataframe with loss values into a dictionary
    metrics_df_evaluation = metrics_df[metrics_df["total_loss"].isna()].dropna(axis=1, how="all")       # The final row is a evaluation row for the accuracy (or something)
    metrics_dict_evaluation = metrics_df_evaluation.to_dict(orient="list")                              # Convert the dataframe with the accuracy results into a dictionary
    for key in metrics_dict_training.keys():                                                            # Looping through all key values in the training metrics dictionary
        if "loss" not in key.lower(): continue                                                          # If the key is not a loss-key, skip to the next key
        key_val, mov_avg_val = list(), list()                                                           # Initiate lists to store the actual values and the moving-average computed values
        for item in metrics_dict_training[key]:                                                         # Loop through each item in the dict[key]->value list
            key_val.append(item)                                                                        # Append the actual item value to the key_val list
            mov_avg_val.append(mov_avg_array(inp_array=key_val, mov_of_last_n_elements=10, output_last_n_elements=1).item())    # Compute the next mov_avg val for the last 10 elements
        metrics_dict_training[key] = mov_avg_val                                                        # Assign the newly computed moving average of the dict[key]->values to the dictionary
    return metrics_dict_training                                                                        # Return the moving average value dictionary
history = load_metrics(config)


# Function to display learning curves
def show_history(history, save_folder=None, display_fig=True):                                          # Define a function to visualize the learning curves
    fig, ax_list = plt.subplots(nrows=1, ncols=4, figsize=(22, 5))                                      # Create a figure and a list of subplot axes
    loss_total = [key for key in history.keys() if "total_loss" in key.lower()]                         # Find all keys with loss_ce
    loss_ce = [key for key in history.keys() if "loss_ce" in key.lower()]                               # Find all keys with loss_ce
    loss_dice = [key for key in history.keys() if "loss_dice" in key.lower()]                           # Find all keys with loss_dice
    loss_mask = [key for key in history.keys() if "loss_mask" in key.lower()]                           # Find all keys with loss_mask
    hist_keys = [loss_total, loss_ce, loss_dice, loss_mask]                                             # Combine the key-lists into a list of lists
    ax_titles = ["total_loss", "loss_ce", "loss_dice", "loss_mask"]                                     # Create titles for the axes
    colors = ["blue", "red", "black", "green"]                                                          # Colors for the line plots
    for ii, ax in enumerate(ax_list):                                                                   # Looping through each subplot axes
        ax.set_xlabel("iteration #")                                                                    # Set correct x-label on the subplot axes
        ax.set_ylabel(ax_titles[ii])                                                                    # Set correct y-label on the subplot axes
        ax.grid("on")                                                                                   # Activate the grid on the subplot axes
        ax.set_xlim(left=0, right=history["iteration"][-1])                                             # Set correct x-limits on the subplot axes
        ax.set_ylim(bottom=0, top=1)                                                                    # Set y-limits on the subplot axes
        y_top_val = 0                                                                                   # Initiate counter to change the top-limit of the y-limits for the subplot axes

        # Plot the points
        for kk, key in enumerate(hist_keys[ii]):                                                        # Looping through all keys in the history dict that will be shown on the current subplot axes
            if np.max(history[key]) > y_top_val:                                                        # If the maximum value in the array is larger than the current y_top_val ...
                y_top_val = np.ceil(np.max(history[key])*10)/10                                         # ... y_top_val is updated and rounded to the nearest 0.1
            ax.plot(history["iteration"], history[key], color=colors[kk], linestyle="-", marker=".")    # Plot the data
        if y_top_val <= 0.05: ax.set_ylim(bottom=-0.05, top=0.05)                                       # If the max y-value is super low, the limits are changed
        else: ax.set_ylim(top=y_top_val)                                                                # Set the final, updated y_top_value as the y-top-limit on the current subplot axes
    if save_folder != None:                                                                             # If a folder to save the figure has been given ...
        fig.savefig(os.path.join(save_folder, "Learning_curves.jpg"), bbox_inches="tight")              # ... the figure is saved in that folder
    if display_fig==False: plt.close(fig)                                                               # If the user chose to not display the figure, the figure is closed
    return fig                                                                                          # The figure handle is returned
fig = show_history(history)

