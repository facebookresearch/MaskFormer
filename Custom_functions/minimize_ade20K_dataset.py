import os 

# Get the folder containing the ADE20K dataset
data_folder = os.path.join("C:\\", "Users", "Nico-", "Documents", "Python_Projects", )
if not os.path.isdir(data_folder):
    data_folder = os.path.join("/mnt", "c", "Users", "Nico-", "Documents", "Python_Projects")
if not os.path.isdir(data_folder):
    data_folder = os.path.join("/mnt", "home_shared", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Datasets")
data_folder = os.path.join(data_folder, "MaskFormer", "datasets", "ADEChallengeData2016")

# Get lists of all the images in the training and validation datasets
ade20K_train_images = [x for x in os.listdir(os.path.join(data_folder, "images", "training")) if x.endswith(".jpg")]
ade20K_val_images = [x for x in os.listdir(os.path.join(data_folder, "images", "validation")) if x.endswith(".jpg")]
ade20K_train_masks = [x for x in os.listdir(os.path.join(data_folder, "annotations_detectron2", "training")) if x.endswith(".png")]
ade20K_val_masks = [x for x in os.listdir(os.path.join(data_folder, "annotations_detectron2", "validation")) if x.endswith(".png")]

