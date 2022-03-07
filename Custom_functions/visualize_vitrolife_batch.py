import os
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
from register_vitrolife_dataset import vitrolife_dataset_function                   # Import function to get the dataset_dictionaries of the vitrolife dataset
from detectron2.data import MetadataCatalog


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
def apply_colormap(mask, split="train"):
    colors_used = list(MetadataCatalog["vitrolife_dataset_"+split].stuff_colors)
    labels_used = list(MetadataCatalog["vitrolife_dataset_"+split].stuff_dataset_id_to_contiguous_id.values())
    color_array = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label_id, label in enumerate(labels_used):
        color_array[mask == label] = colors_used[label_id]
    return color_array


# Define function to plot the images
def visualize_the_images(data_split="train", num_images=5, figsize=(16, 8), position=[0.10, 0.09, 0.80, 0.75]):
    # Extract information about the vitrolife dataset
    dataset_dicts = vitrolife_dataset_function(data_split)
    num_rows, num_cols = 2, num_images
    fig = plt.figure(figsize=figsize)
    for img_idx, data_dict in enumerate(random.sample(dataset_dicts, num_images)):
        # Plot the image
        img = cv2.cvtColor(cv2.imread(data_dict["file_name"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        plt.subplot(num_rows, num_cols, img_idx+1)
        plt.title("Image #{} from {:s} split with {:.0f} PN".format(img_idx+1, data_split, data_dict["image_custom_info"]["PN_image"]))
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        # Plot the mask
        mask = cv2.imread(data_dict["sem_seg_file_name"], cv2.IMREAD_GRAYSCALE)
        plt.subplot(num_rows, num_cols, num_cols+img_idx+1)
        plt.title("Mask #{} from {:s} split with {:.0f} PN".format(img_idx+1, data_split, data_dict["image_custom_info"]["PN_image"]))
        plt.imshow(apply_colormap(mask, split=data_split), cmap="gray")
        plt.axis("off")
    try: fig = move_figure_position(fig=fig, position=position)
    except: pass
    fig.show()
    
    return fig


