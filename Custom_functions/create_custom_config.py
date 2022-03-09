import os                                                                                   # Used to navigate around the current os and folder structure
import torch                                                                                # torch is implemented to check if a GPU is available
import numpy as np
from sys import path as sys_PATH                                                            # Import the PATH variable
from register_vitrolife_dataset import register_vitrolife_data_and_metadata_func            # Import function to register the vitrolife datasets in Detectron2 
from detectron2.data import MetadataCatalog                                                 # Catalog containing metadata for all datasets available in Detectron2
from detectron2.config import get_cfg                                                       # Function to get the default configuration from Detectron2
from detectron2.projects.deeplab import add_deeplab_config                                  # Used to merge the default config with the deeplab config before training
from mask_former import add_mask_former_config                                              # Used to add the new configuration to the list of possible configurations

class Namespace(object): pass                                                               # Used to pass the FLAG argument as an empty namespace
# Define function to get all keys in a nested dictionary
def accumulate_keys(dct):
    key_list = list()
    def accumulate_keys_recursive(dct):
        for key in dct.keys():
            if isinstance(dct[key], dict): accumulate_keys_recursive(dct[key])
            else: key_list.append(key.upper())
    accumulate_keys_recursive(dct)
    return key_list

# Define a function to create a custom configuration in the chosen config_dir and takes a namespace option
def createVitrolifeConfiguration(FLAGS):
    # Register the vitrolife datasets
    register_vitrolife_data_and_metadata_func(debugging=FLAGS.debugging)
    assert any(["vitrolife" in x for x in list(MetadataCatalog)]), "Datasets have not been registered correctly"


    # Locate the folder containing other configurations
    MaskFormer_dir = [x for x in sys_PATH if x.endswith("MaskFormer")][0]
    config_folder = os.path.join(MaskFormer_dir, "configs", "ade20k-150")

    # Get all keys from the FLAGS input argument
    key_list = accumulate_keys(vars(FLAGS))

    # Alter the configuration and make it as wanted. Not all keys will be used, e.g. Panoptic keys are unused for now
    # cfg = CN(CN.load_yaml_with_base(os.path.join(config_folder, "maskformer_R50_bs16_160k.yaml"), allow_unsafe=True))
    cfg = get_cfg()                                                                         # Get the default configuration from detectron2.
    add_deeplab_config(cfg)                                                                 # Add some deeplab (i.e. sem_seg) config values
    add_mask_former_config(cfg)                                                             # Add some default values used for semantic segmentation to the config and choose datasetmapper
    cfg.merge_from_file(os.path.join(config_folder, "swin", "maskformer_swin_small_bs16_160k.yaml"))    # Merge with the config for the small swin transformer
    cfg.merge_from_file(os.path.join(config_folder, "maskformer_R50_bs16_160k.yaml"))       # Merge with the small maskformer config
    cfg.merge_from_file(os.path.join(config_folder, "Base-ADE20K-150.yaml"))                # Merge with the base config for ade20K dataset
    cfg.SOLVER.BASE_LR = FLAGS.learning_rate if "LEARNING_RATE" in key_list else 1e-3       # Starting learning rate
    cfg.SOLVER.IMS_PER_BATCH = FLAGS.batch_size if "BATCH_SIZE" in key_list else 1          # Batch size used when training => batch_size pr GPU = batch_size // num_gpus
    cfg.SOLVER.MAX_ITER = FLAGS.max_iter if "MAX_ITER" in key_list else int(2e4)            # Maximum number of iterations to train for
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"                                      # Default learning rate scheduler
    cfg.SOLVER.NESTEROV = True                                                              # Whether or not the learning algorithm will use Nesterow momentum
    cfg.SOLVER.WEIGHT_DECAY = float(2e-5)                                                   # A small lambda value for the weight decay
    cfg.TEST.AUG.FLIP = False                                                               # No random flipping or augmentation used for inference
    cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED = False                                          # Disable the panoptic head during inference
    cfg.DATALOADER.NUM_WORKERS = FLAGS.Num_workers if "NUM_WORKERS" in key_list else 2      # Set the number of workers to only 2
    cfg.INPUT.CROP.ENABLED =  FLAGS.Crop_Enabled if "CROP_ENABLED" in key_list else False   # We will not allow any cropping of the input images
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'                       # Assign the device on which the model should run
    cfg.MODEL.RESNETS.DEPTH = FLAGS.Resnet_Depth if "RESNET_DEPTH" in key_list else 50      # Assign the depth of the backbone feature extracting model
    cfg.MODEL.WEIGHTS = FLAGS.Model_weights if "MODEL_WEIGHTS" in key_list else ""          # Whether or not to start with randomly initialized weights or just an earlier checkpoint
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 2                                                   # Set the weight for the dice loss
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 10                                                  # Set the weight for the mask loss
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5                                           # Assign the threshold used for the model
    config_name = "config_initial.yaml"                                                     # Initial name for the configuration that will be saved in the cfg.OUTPUT_DIR
    if "DATASET_NAME" in key_list:                                                          # If we are choosing a dataset using the command line input arguments ...
        if "vitrolife" in FLAGS.dataset_name.lower():                                       # ... and the default value of vitrolife is chosen ...
            cfg["DATASETS"]["TRAIN"] = ("vitrolife_dataset_train",)                         # ... define the training dataset by using the config as a dictionary
            cfg.DATASETS.TEST = ("vitrolife_dataset_val",)                                  # ... define the validation dataset by using the config as a CfgNode 
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(MetadataCatalog[cfg.DATASETS.TEST[0]].stuff_classes)   # Assign the number of classes for the model to segment
            cfg.INPUT.FORMAT = "BGR"                                                        # The input format is set to be BGR, like the visualization method
            cfg.INPUT.MIN_SIZE_TRAIN = FLAGS.Img_size_min if "IMG_SIZE_MIN" in key_list else 500    # The minimum size length for one side of the training images
            cfg.INPUT.MAX_SIZE_TRAIN = FLAGS.Img_size_max if "IMG_SIZE_MAX" in key_list else 500    # The maximum size length for one side of the training images
            cfg.INPUT.MIN_SIZE_TEST = FLAGS.Img_size_min if "IMG_SIZE_MIN" in key_list else 500     # The minimum size length for one side of the validation images
            cfg.INPUT.MAX_SIZE_TEST = FLAGS.Img_size_max if "IMG_SIZE_MAX" in key_list else 500     # The maximum size length for one side of the validation images
            cfg.MODEL.PIXEL_MEAN = [100.15, 102.03, 103.89]                                 # Write the correct image mean value for the entire vitrolife dataset
            cfg.MODEL.PIXEL_STD = [57.32, 59.69, 61.93]                                     # Write the correct image standard deviation value for the entire vitrolife dataset
            cfg.OUTPUT_DIR = os.path.join(MaskFormer_dir, "output_vitrolife_"+FLAGS.output_dir_postfix) # Get MaskFormer directory and name the output directory
            cfg.SOLVER.CHECKPOINT_PERIOD = MetadataCatalog[cfg.DATASETS.TRAIN[0]].num_files_in_dataset  # Save a new model checkpoint after each epoch, i.e. after everytime the entire trainining set has been seen by the model
            cfg.TEST.EVAL_PERIOD = MetadataCatalog[cfg.DATASETS.TEST[0]].num_files_in_dataset           # Evaluation after each epoch. Thus in the logs it can be seen which iteration was "best" and then that checkpoint can be loaded later
            cfg.SOLVER.STEPS = np.subtract([int(x+1)*np.min([50, cfg.SOLVER.MAX_ITER]) for x in range(20)],1).tolist()  # The warm up steps for the learning rate scheduler. Steps has to be smaller than max_iter
            config_name = "vitrolife_" + config_name                                        # Prepend the config name with "vitrolife"
    if "DEBUGGING" in key_list:                                                             # Checking if debugging state is an option
        if FLAGS.debugging==True:                                                           # If we are debugging the model ...
            cfg.SOLVER.CHECKPOINT_PERIOD = int(np.subtract(cfg.SOLVER.MAX_ITER, 1))         # ... a checkpoint will only be saved after the final iteration
            cfg.TEST.EVAL_PERIOD = int(np.subtract(cfg.SOLVER.MAX_ITER, 1))                 # ... inference will only happen after the final iteration
            cfg.DATASETS.TEST = cfg.DATASETS.TRAIN                                          # ... and inference will only happen on the training dataset

    # Write the new config as a .yaml file - it already does, in the output dir...
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)                                              # Create the output folder, if it doesn't already exist
    with open(os.path.join(cfg.OUTPUT_DIR, config_name), "w") as f:                         # Open a object instance with the config file
        f.write(cfg.dump())                                                                 # Dump the configuration to a file named config_name in cfg.OUTPUT_DIR
    f.close()
    
    # Return the custom configuration
    return cfg

