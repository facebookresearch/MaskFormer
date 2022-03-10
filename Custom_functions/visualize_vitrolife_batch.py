import os
import re
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from register_vitrolife_dataset import vitrolife_dataset_function                   # Import function to get the dataset_dictionaries of the vitrolife dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper, build_detection_train_loader
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
def apply_colormap(mask, config):
    colors_used = list(MetadataCatalog[config.DATASETS.TEST[0]].stuff_colors)       # Read the colors used in the Metadatacatalog. If no colors are assigned, random colors are used
    if "vitrolife" in config.DATASETS.TEST[0].lower():                              # If we are working on the vitrolife dataset ...
        labels_used = list(MetadataCatalog[config.DATASETS.TEST[0]].stuff_dataset_id_to_contiguous_id.values()) # ... labels_used will be read from the MetadataCatalog
    else: labels_used = list(range(1, 1+len(MetadataCatalog["ade20k_sem_seg_train"].stuff_classes)))    # Else, labels is just 1:num_classes
    color_array = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)       # Allocate a RGB 3D array of zeros
    for label_idx, label in enumerate(labels_used):                                 # Loop through each label from the labels_used found from the MetadataCatalog
        color_array[mask == label] = colors_used[label_idx]                         # Assign all pixels in the mask with the current label_value the colors_used[idx] value
    return color_array                                                              # Return the colored mask


# Define a function to extract numbers from a string
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
    model_files = [x for x in os.listdir(config.OUTPUT_DIR) if "model" in x.lower() and x.endswith(".pth") and not np.isnan(extractNumbersFromString(x))]   # Find all saved model checkpoints
    if len(model_files) >= 1:                                                       # If any model checkpoint is found, 
        iteration_numbers = [extractNumbersFromString(x, int) for x in model_files] # Find the iteration numbers for when they were saved
        latest_iteration_idx = np.argmax(iteration_numbers)                         # Find the index of the model checkpoint with the latest iteration number
        config.MODEL.WEIGHTS = os.path.join(config.OUTPUT_DIR, model_files[latest_iteration_idx])   # Assign the latest model checkpoint to the config
        for model_file in model_files:                                              # Loop through all found model checkpoint files
            if os.path.join(config.OUTPUT_DIR,model_file) != config.MODEL.WEIGHTS:  # If the current model_file is not the checkpoint file ...
                os.remove(os.path.join(config.OUTPUT_DIR,model_file))               # ... remove the current model_file
    return config                                                                   # Return the updated config


# Define a function to predict some label-masks for the dataset
def create_batch_img_ytrue_ypred(config, data_split, FLAGS, data_batch=None):
    config = putModelWeights(config)                                                # Change the config and append the latest model as the used checkpoint
    predictor = DefaultPredictor(cfg=config)                                        # Create a default predictor instance
    Softmax_module = nn.Softmax(dim=2)                                              # Create a module to compute the softmax value along the final, channel, dimension of the predicted images
    if data_batch == None:                                                          # If no batch with data was send to the function ...
        if "vitrolife" in FLAGS.dataset_name.lower():                               # ... and if we are using the vitrolife dataset
            dataset_dicts = vitrolife_dataset_function(data_split, debugging=FLAGS.debugging)   # ... the list of dataset_dicts from vitrolife is computed.
        else: dataset_dicts = DatasetCatalog.get("ade20k_sem_seg_{:s}".format(data_split))  # Else we use the ADE20K dataset
        if len(dataset_dicts) < FLAGS.num_images: FLAGS.num_images = len(dataset_dicts)     # If we are debugging, we'll only return 2 images from vitrolife dataset, thus num_images must be lowered
        dataloader = build_detection_train_loader(dataset_dicts, mapper=DatasetMapper(putModelWeights(config),  # Create a dataloader with the list of dictionaries using the default mapper (maps from filename ...
        is_train=False), total_batch_size=FLAGS.num_images)                         #  to files, i.e. performs augmentation etc) specifying the batch_size to be drawn
        data_batch = next(iter(dataloader))                                         # Extract the next batch from the dataloader
    img_ytrue_ypred = {"input": list(), "y_pred": list(), "y_true": list(), "PN": list()}   # Initiate a dictionary to store the input images, ground truth masks and the predicted masks
    for data in data_batch:                                                         # Iterate over each data sample in the batch from the dataloader
        img = torch.permute(data["image"], (1,2,0)).numpy()                         # Input image [H,W,C]
        y_true = data["sem_seg"].numpy()                                            # Ground truth label mask [H,W]
        y_true_col = apply_colormap(mask=y_true, config=config)                     # Ground truth color mask
        out = predictor.__call__(img)                                               # Predicted output dictionary. The call function needs images in BGR format.
        out_img = torch.permute(out["sem_seg"], (1,2,0))                            # Predicted output image [H,W,C]
        out_img_softmax = Softmax_module(out_img)                                   # Softmax of predicted output image
        y_pred = torch.argmax(out_img_softmax,2).cpu()                              # Predicted output image [H,W]
        y_pred_col = apply_colormap(mask=y_pred, config=config)                     # Predicted colormap for predicted output image
        # Append the input image, y_true and y_pred to the dictionary
        img_ytrue_ypred["input"].append(img)                                        # Append the input image to the dictionary
        img_ytrue_ypred["y_true"].append(y_true_col)                                # Append the ground truth to the dictionary
        img_ytrue_ypred["y_pred"].append(y_pred_col)                                # Append the predicted mask to the dictionary
        if "vitrolife" in FLAGS.dataset_name.lower():                               # If we are visualizing the vitrolife dataset
            img_ytrue_ypred["PN"].append(int(data["image_custom_info"]["PN_image"]))# Read the true number of PN on the current image
    return img_ytrue_ypred, data_batch, FLAGS


# Define function to plot the images
def visualize_the_images(config, FLAGS, position=[0.55, 0.08, 0.40, 0.75], data_batch=None):
    # Get the datasplit and number of images to show
    data_split = "train" if FLAGS.debugging else config.DATASETS.TEST[0].split("_")[-1] # Split the dataset name at all the '_' and extract the final part, i.e. the datasplit
    before_train = True if data_batch == None and data_split != "test" else False       # The images are visualized before starting training, if the data_batch is None. Else training has been completed.
    
    # Extract information about the dataset used
    img_ytrue_ypred, data_batch, FLAGS = create_batch_img_ytrue_ypred(config=config,# Create the batch of images that needs to be visualized
            data_split=data_split, FLAGS=FLAGS, data_batch=data_batch)              # And return the images in the data_batch dictionary
    num_rows, num_cols = 3, FLAGS.num_images                                        # The figure will have three rows (input, y_pred, y_true) and one column per image
    fig = plt.figure(figsize=(FLAGS.num_images*4.5, 8))                             # Create the figure object
    row = 0                                                                         # Initiate the row index counter (all manual indexing could have been avoided by having created img_ytrue_ypred as an OrderedDict)
    for key in img_ytrue_ypred.keys():                                              # Loop through all the keys in the batch dictionary
        if key.lower() not in ['input', 'y_true', 'y_pred']: continue               # If the key is not one of (input, y_pred, y_true), we simply skip to the next one
        for col, img in enumerate(img_ytrue_ypred[key]):                            # Loop through all available images in the dictionary
            plt.subplot(num_rows, num_cols, row*num_cols+col+1)                     # Create the subplot instance
            plt.axis("off")                                                         # Remove axis tickers
            if "vitrolife" in FLAGS.dataset_name.lower():                           # If we are visualizing the vitrolife dataset
                plt.title("{:s} with {:.0f} PN".format(key, img_ytrue_ypred["PN"][col]))# Create the title for the plot with the number of PN
            else: plt.title("{:s}".format(key))                                     # Otherwise simply put the key, i.e. either input, y_pred or y_true.
            plt.imshow(img, cmap="gray")                                            # Display the image
        row += 1                                                                    # Increase the row counter by 1
    try: fig = move_figure_position(fig=fig, position=position)                     # Try and move the figure to the wanted position (only possible on home computer with a display)
    except: pass                                                                    # Except, simply just let the figure retain the current position
    fig.tight_layout()                                                              # Assures the subplots are plotted tight around each other
    figure_name = "Segmented_{:s}_data_samples_from_{:s}_training.jpg".format(data_split, "before" if before_train else "after")    # Create a name for the figure
    fig.savefig(os.path.join(config.OUTPUT_DIR, figure_name), bbox_inches="tight")  # Save the figure in the output directory
    return fig, data_batch, putModelWeights(config), FLAGS                          # Return the figure, the dictionary with the used images and the updated config with a new model checkpoint


