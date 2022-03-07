from doctest import FAIL_FAST
import numpy as np 
import os
import glob
from PIL import Image
import pandas as pd
import os
from detectron2.data import DatasetCatalog, MetadataCatalog

# Define the function to return the list of dictionaries with information regarding all images available in the vitrolife dataset
def vitrolife_dataset_function(run_mode="train", debugging=False):
    # Find the folder containing the vitrolife dataset
    vitrolife_dataset_filepath = os.path.join(os.environ["DETECTRON2_DATASETS"], "Vitrolife_dataset")
    
    # Find the metadata file
    metadata_file = os.path.join(vitrolife_dataset_filepath, "metadata.csv")
    df_data = pd.read_csv(metadata_file)
    df_data = df_data.set_index(["HashKey","Well"])

    # Create the list of dictionaries with information about all images
    img_mask_pair_list = []                                                                 # Initiate the list to store the information about all images
    count = 0                                                                               # Initiate a counter to count the number of images inserted to the dataset
    for img_filename in os.listdir(os.path.join(vitrolife_dataset_filepath, "raw_images")): # Loop through all files in the raw_images folder
        img_filename_wo_ext = os.path.splitext(os.path.basename(img_filename))[0]           # Get the image filename without .jpg extension
        img_filename_wo_ext_parts = img_filename_wo_ext.split("_")                          # Split the filename where the _ is
        hashkey = img_filename_wo_ext_parts[0]                                              # Extract the hashkey from the filename
        well = int(img_filename_wo_ext_parts[1][1:])                                        # Extract the well from the filename
        row = df_data.loc[hashkey,well]                                                     # Find the row of the corresponding file in the dataframe
        data_split = row["split"]                                                           # Find the split for the current image, i.e. either train, val or test
        if data_split != run_mode: continue                                                 # If the current image is supposed to be in another split, then continue to the next image
        mask_filename = glob.glob(os.path.join(vitrolife_dataset_filepath, 'masks',img_filename_wo_ext + '*'))  # Find the corresponding mask filename
        if len(mask_filename) != 1: continue                                                # Continue only if we find only one mask filename
        mask_filename = os.path.basename(mask_filename[0])                                  # Extract the mask filename from the list
        width_img, height_img = Image.open(os.path.join(vitrolife_dataset_filepath, "raw_images", img_filename)).size   # Get the image size of the img_file
        width_mask, height_mask = Image.open(os.path.join(vitrolife_dataset_filepath, "masks", mask_filename)).size     # Get the mask size of the mask_file
        if not all([width_img==width_mask, height_img==height_mask]): continue              # The image and mask have to be of the same size
        current_pair = {"file_name": os.path.join(vitrolife_dataset_filepath, "raw_images", img_filename),      # Initiate the dict of the current image with the full filepath + filename
                        "height": height_img,                                               # Write the image height
                        "width": width_img,                                                 # Write the image width
                        "image_id": img_filename_wo_ext,                                    # A unique key for the current image
                        "sem_seg_file_name": os.path.join(vitrolife_dataset_filepath, "masks", mask_filename),  # The full filepath + filename for the mask ground truth label image
                        "image_custom_info": row}                                           # Add all the info from the current row to the dataset
        img_mask_pair_list.append(current_pair)                                             # Append the dictionary for the current pair to the list of images for the given dataset
        count += 1                                                                          # Increase the sample counter 
        if count >= 20 and debugging==True: break                                           # When debugging, we will only use 20 samples in both train, val and test
    
    assert len(img_mask_pair_list) >= 1, print("No image/mask pairs found in {} subfolders 'raw_image' and 'masks'".format(vitrolife_dataset_filepath))
    return img_mask_pair_list

# Function to register the dataset and the meta dataset for each of the three splits, [train, val, test]
def register_vitrolife_data_and_metadata_func(debugging=False):
    class_labels = ["Background", "Well", "Zona", "Perivitelline space", "Cell", "PN"]
    stuff_id = {ii: ii for ii in range(len(class_labels))}
    stuff_colors = [(0,0,0), (255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255)]
    for split_mode in ["train", "val", "test"]:
        DatasetCatalog.register("vitrolife_dataset_{:s}".format(split_mode), lambda split_mode=split_mode: vitrolife_dataset_function(run_mode=split_mode, debugging=debugging))
        MetadataCatalog.get("vitrolife_dataset_{:s}".format(split_mode)).set(stuff_classes=class_labels,
                                                                            stuff_colors = stuff_colors,
                                                                            stuff_dataset_id_to_contiguous_id = stuff_id,
                                                                            ignore_label=255,
                                                                            evaluator_type="sem_seg",
                                                                            num_files_in_dataset=len(DatasetCatalog["vitrolife_dataset_{:}".format(split_mode)]()))

# Test that the function will actually return a list of dicts
img_mask_list_train = vitrolife_dataset_function(run_mode="train")
img_mask_list_val = vitrolife_dataset_function(run_mode="val")
img_mask_list_test = vitrolife_dataset_function(run_mode="test")


