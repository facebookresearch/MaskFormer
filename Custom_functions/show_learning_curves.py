# Import libraries
import os
from turtle import color 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


# Function to display learning curves
def show_history(config, display_fig=True):                                                             # Define a function to visualize the learning curves
    history = load_metrics(config=config)                                                               # Load the metrics into the history dictionary
    loss_total = [key for key in history.keys() if "total_loss" in key.lower()]                         # Find all keys with loss_ce
    loss_ce = [key for key in history.keys() if "loss_ce" in key.lower()]                               # Find all keys with loss_ce
    loss_dice = [key for key in history.keys() if "loss_dice" in key.lower()]                           # Find all keys with loss_dice
    loss_mask = [key for key in history.keys() if "loss_mask" in key.lower()]                           # Find all keys with loss_mask
    learn_rate = [key for key in history.keys() if "lr" in key.lower()]                                 # Find all keys with loss_mask
    hist_keys = [loss_total, learn_rate, loss_ce, loss_dice, loss_mask]                                 # Combine the key-lists into a list of lists
    ax_titles = ["total_loss", "learning_rate", "loss_ce", "loss_dice", "loss_mask"]                    # Create titles for the axes
    colors = ["blue", "red", "black", "green", "magenta", "cyan", "yellow"]                             # Colors for the line plots
    fig = plt.figure(figsize=(17,8))                                                                    # Create the figure
    n_rows, n_cols, ax_count = 2, (2,3), 0                                                              # Initiate values for the number of rows and columns
    for row in range(n_rows):                                                                           # Loop through all rows
        for col in range(n_cols[row]):                                                                  # Loop through all columns in the current row
            plt.subplot(n_rows, n_cols[row], 1+row*n_cols[row]+col)                                     # Create a new subplot
            plt.xlabel(xlabel="Iteration #")                                                            # Set correct xlabel
            plt.ylabel(ylabel=ax_titles[ax_count].replace("_", " "))                                    # Set correct ylabel
            plt.grid(True)                                                                              # Activate the grid on the plot
            plt.xlim(left=0, right=np.max(history["iteration"]))                                        # Set correct xlim
            plt.title(label=ax_titles[ax_count].replace("_", " ").capitalize())                         # Set plot title
            y_top_val = 0                                                                               # Initiate a value to determine the y_max value of the plot
            for kk, key in enumerate(hist_keys[ax_count]):                                              # Looping through all keys in the history dict that will be shown on the current subplot axes
                if np.max(history[key]) > y_top_val:                                                    # If the maximum value in the array is larger than the current y_top_val ...
                    y_top_val = np.ceil(np.max(history[key])*10)/10                                     # ... y_top_val is updated and rounded to the nearest 0.1
                plt.plot(history["iteration"], history[key], color=colors[kk], linestyle="-", marker=".")   # Plot the data
            plt.legend([key for key in hist_keys[ax_count]], framealpha=0.5)                            # Create a legend for the subplot with the history keys displayed
            ax_count += 1                                                                               # Increase the subplot counter
        if y_top_val <= 0.05 and "lr" not in key.lower(): plt.ylim(bottom=-0.05, top=0.05)              # If the max y-value is super low, the limits are changed
        else: plt.ylim(top=y_top_val)                                                                   # Set the final, updated y_top_value as the y-top-limit on the current subplot axes
        if "lr" in key.lower():                                                                         # If we are plotting the learning rate ...
            plt.ylim(bottom=np.min(history[key]) / 2, top=np.ceil(np.max(history[key])*100)/100)        # ... the y_limits are rounded to the nearest 0.01
            plt.yscale('log')                                                                           # ... the y_scale will be logarithmic
    try: fig.savefig(os.path.join(config.OUTPUT_DIR, "Learning_curves.jpg"), bbox_inches="tight")       # Try and save the figure in the OUTPUR_DIR ...
    except: pass                                                                                        # ... otherwise simply skip saving the figure
    if display_fig==False: plt.close(fig)                                                               # If the user chose to not display the figure, the figure is closed
    fig.tight_layout()                                                                                  # Make the figure tight_layout, which assures the subplots will be better spaced together
    return fig                                                                                          # The figure handle is returned