import os
import re
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from register_vitrolife_dataset import vitrolife_dataset_function                   # Import function to get the dataset_dictionaries of the vitrolife dataset
from detectron2.data import MetadataCatalog, DatasetMapper, build_detection_train_loader
from detectron2.engine.defaults import DefaultPredictor

# Move the figure to the wanted position when displaying
try:
    import pyautogui
    def move_figure_position(fig=plt.figure(), screensize=list(pyautogui.size()),   # Define a function to move a figure ...
                            dpi=100, position=[0.10, 0.09, 0.80, 0.75]):            # ... to a specified position on the screen
        fig = plt.figure(fig)                                                       # Make the wanted figure the current figure again
        # screensize[1] = np.round(np.divide(screensize[1], 1.075))                 # Reduce height resolution as the processbar in the bottom is part of the screen size
        screensize_inches = np.divide(screensize,dpi)                               # Convert the screensize into inches
        fig.set_figheight(position[3] * screensize_inches[1])                       # Set the wanted height of the figure
        fig.set_figwidth(position[2] * screensize_inches[0])                        # Set the wanted width of the figure
        figManager = plt.get_current_fig_manager()                                  # Get the current manager (i.e. window execution commands) of the current figure
        upper_left_corner_position = "+{:.0f}+{:.0f}".format(                       # Define a string with the upper left corner coordinates ...
            screensize[0]*position[0], screensize[1]*position[1])                   # ... which are read from the position inputs
        figManager.window.wm_geometry(upper_left_corner_position)                   # Move the figure to the upper left corner coordinates
        return fig                                                                  # Return the figure handle
except: pass


# Define function to apply a colormap on the images
def apply_colormap(mask, split):
    colors_used = list(MetadataCatalog["vitrolife_dataset_"+split].stuff_colors)
    labels_used = list(MetadataCatalog["vitrolife_dataset_"+split].stuff_dataset_id_to_contiguous_id.values())
    color_array = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label_id, label in enumerate(labels_used):
        color_array[mask == label] = colors_used[label_id]
    return color_array

def extractNumbersFromString(str, dtype=float, numbersWanted=1):
    try: vals = dtype(str)                                                          # At first, simply try to convert the string into the wanted dtype
    except:                                                                         # Else, if that is not possible ...
        vals = [float(s) for s in re.findall(r"[-+]?\d*\.\d+|\d+", str)]            # Extract all the numbers from the string and put them in a list
        if len(vals) > 0:                                                           # If any numbers is found ...
            for kk in range(len(vals)):                                             # Loop through all the found numbers
                vals[kk] = dtype(vals[kk])                                          # Convert each of the found numbers into the wanted dtype
                if kk == numbersWanted-1: break                                     # If we have convert all the numbers wanted, we'll stop the loop
            vals = vals[:numbersWanted]                                             # Then we'll only use up to 'numbersWanted' found numbers
            if numbersWanted==1: vals = vals[0]                                     # If we only want 1 number, then we'll extract that from the list
        else: vals = np.nan                                                         # ... else if no numbers were found, return NaN
    return vals                                                                     # Return the wanted numbers, either as a type 'dtype' or, if multiple numbers, a list of 'dtypes'

# Define a function to put the latest saved model as the model_weights in the config before creating the dataloader
def putModelWeights(config):
    model_files = [x for x in os.listdir(config.OUTPUT_DIR) if "model" in x and x.endswith(".pth") and not np.isnan(extractNumbersFromString(x))]
    if len(model_files) >= 1:
        iteration_numbers = [extractNumbersFromString(x, int) for x in model_files]
        latest_iteration_idx = np.argmax(iteration_numbers)
        config.MODEL.WEIGHTS = os.path.join(config.OUTPUT_DIR, model_files[latest_iteration_idx])
    return config

# Define a function to convert a dictionary with filenames into a img_sem_seg batched dictionary like the output from the dataloader
def filename_dict_to_datalist(filename_dict):
    dataset_list = list()
    for img_path, y_true_path, PN, img_info in zip(filename_dict["image"], filename_dict["sem_seg"], filename_dict["PN"], filename_dict["image_custom_info"]):
        current_files_dict = {}
        current_files_dict["image"] = torch.permute(torch.from_numpy(cv2.imread(img_path, cv2.IMREAD_COLOR)), (2,0,1))
        current_files_dict["sem_seg"] = torch.from_numpy(cv2.imread(y_true_path, cv2.IMREAD_GRAYSCALE))
        current_files_dict["PN"] = PN
        current_files_dict["image_custom_info"] = img_info
        dataset_list.append(current_files_dict)
    return dataset_list

# Define a function to predict some label-masks for the dataset
def create_batch_Img_ytrue_ypred(config, data_split, num_images, filename_dict):
    predictor = DefaultPredictor(cfg=config)
    Softmax_module = nn.Softmax(dim=2)
    if filename_dict == None:
        dataset_dicts = vitrolife_dataset_function(data_split, debugging=True)
        dataloader = build_detection_train_loader(dataset_dicts, mapper=DatasetMapper(putModelWeights(config), is_train=False), total_batch_size=num_images)
        data_batch = next(iter(dataloader))
    else:
        data_batch = filename_dict_to_datalist(filename_dict)
    filename_dict = {"image": list(), "sem_seg": list(), "PN": list(), "image_custom_info": list()}
    img_ytrue_ypred = {"input": list(), "y_pred": list(), "y_true": list(), "PN": list()}
    for data in data_batch:
        img = torch.permute(data["image"], (1,2,0)).numpy()                         # Input image [H,W,C]
        y_true = data["sem_seg"].numpy()                                            # Ground truth label mask [H,W]
        y_true_col = apply_colormap(mask=y_true, split=data_split)                  # Ground truth color mask
        out = predictor.__call__(img)                                               # Predicted output dictionary
        out_img = torch.permute(out["sem_seg"], (1,2,0))                            # Predicted output image [H,W,C]
        out_img_softmax = Softmax_module(out_img)                                   # Softmax of predicted output image
        y_pred = torch.argmax(out_img_softmax,2).cpu()                              # Predicted output image [H,W]
        y_pred_col = apply_colormap(mask=y_pred, split=data_split)                  # Predicted colormap for predicted output image
        # Append the input image, y_true and y_pred to the dictionary
        img_ytrue_ypred["input"].append(img)                                        # Append the input image to the dictionary
        img_ytrue_ypred["y_true"].append(y_true_col)                                # Append the ground truth to the dictionary
        img_ytrue_ypred["y_pred"].append(y_pred_col)                                # Append the predicted mask to the dictionary
        img_ytrue_ypred["PN"].append(int(data["image_custom_info"]["PN_image"]))    # Read the true number of PN on the current image
        # Append the filenames to the filename_list
        filename_dict["image"].append(data["image_custom_info"]["img_file"])        # ... the current image filename is entered
        filename_dict["sem_seg"].append(data["image_custom_info"]["mask_file"])     # ... the current ground truth filename is entered
        filename_dict["PN"].append(img_ytrue_ypred["PN"][-1])                       # ... and the current number of PN is entered
        filename_dict["image_custom_info"].append(data["image_custom_info"])
    return img_ytrue_ypred, filename_dict


# Define function to plot the images
def visualize_the_images(config, data_split="train", num_images=5, figsize=(16, 8), position=[0.10, 0.09, 0.80, 0.75], filename_dict=None):
    # Extract information about the vitrolife dataset
    img_ytrue_ypred, filename_dict = create_batch_Img_ytrue_ypred(config=config, data_split=data_split, num_images=num_images, filename_dict=filename_dict)
    num_rows, num_cols = 3, num_images
    fig = plt.figure(figsize=figsize)
    for row, key in enumerate(img_ytrue_ypred.keys()):
        if "pn" in key.lower(): continue
        for col, img in enumerate(img_ytrue_ypred[key]):
            plt.subplot(num_rows, num_cols, row*num_cols+col+1)
            plt.axis("off")
            plt.title("{:s} with {:.0f} PN".format(key, img_ytrue_ypred["PN"][col]))
            plt.imshow(img, cmap="gray")
    try: fig = move_figure_position(fig=fig, position=position)
    except: pass
    fig.tight_layout()
    return fig, filename_dict


