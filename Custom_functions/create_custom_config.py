import os
from pickle import FALSE                                                                                   # Used to navigate around the current os and folder structure
from sys import path as sys_PATH                                                            # Used to get the system PATH variable
from register_vitrolife_dataset import register_vitrolife_data_and_metadata_func            # Import function to register the vitrolife datasets in Detectron2 
from detectron2.data import MetadataCatalog                                                 # Catalog containing metadata for all datasets available in Detectron2
from detectron2.config import get_cfg                                                       # Function to get the default configuration from Detectron2
from detectron2.projects.deeplab import add_deeplab_config                                  # Used to merge the default config with the deeplab config before training
import torch                                                                                # torch is implemented to check if a GPU is available
from mask_former import add_mask_former_config                                              # Used to add the new configuration to the list of possible configurations
from pathlib import Path                                                                    # Used to get parent folders and determine output folder for training
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
    cfg["DATASETS"]["TRAIN"] = ("vitrolife_dataset_train",)                                 # Define the training dataset by using the config as a dictionary
    cfg.DATASETS.TEST = ("vitrolife_dataset_val",)                                          # Define the validation dataset by using the config as a CfgNode 
    if "debugging" in key_list:                                                             # If we are debugging ...
         if FLAGS.debugging==True: ("vitrolife_dataset_train",)                             # ... we will perform the evaluation on the train dataset instead of validation dataset...
    cfg.DATALOADER.NUM_WORKERS = FLAGS.Num_workers if "NUM_WORKERS" in key_list else 2      # Set the number of workers to only 2
    cfg.INPUT.CROP.ENABLED =  FLAGS.Crop_Enabled if "CROP_ENABLED" in key_list else False   # We will not allow any cropping of the input images
    cfg.INPUT.FORMAT = "RGB"                                                                # The input format is set to be RGB
    cfg.INPUT.MIN_SIZE_TRAIN = FLAGS.Img_size_min if "IMG_SIZE_MIN" in key_list else 500    # The minimum size length for one side of the training images
    cfg.INPUT.MAX_SIZE_TRAIN = FLAGS.Img_size_max if "IMG_SIZE_MAX" in key_list else 500    # The maximum size length for one side of the training images
    cfg.INPUT.MIN_SIZE_TEST = FLAGS.Img_size_min if "IMG_SIZE_MIN" in key_list else 500     # The minimum size length for one side of the validation images
    cfg.INPUT.MAX_SIZE_TEST = FLAGS.Img_size_max if "IMG_SIZE_MAX" in key_list else 500     # The maximum size length for one side of the validation images
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'                       # Assign the device on which the model should run
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(MetadataCatalog["vitrolife_dataset_train"].stuff_classes)  # Assign the number of classes for the model to segment
    cfg.MODEL.RESNETS.DEPTH = FLAGS.Resnet_Depth if "RESNET_DEPTH" in key_list else 50      # Assign the depth of the backbone feature extracting model
    cfg.MODEL.WEIGHTS = FLAGS.Model_weights if "MODEL_WEIGHTS" in key_list else ""          # Whether or not to start with randomly initialized weights or just a earlier checkpoint
    cfg.MODEL.PIXEL_MEAN = [100.15, 102.03, 103.89]                                         # Write the correct image mean value for the entire vitrolife dataset
    cfg.MODEL.PIXEL_STD = [57.32, 59.69, 61.93]                                             # Write the correct image standard deviation value for the entire vitrolife dataset
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 5
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 10
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5                                           # Assign the threshold used for the model
    os.makedirs(os.path.join(Path(config_folder).parents[1], "output_vitrolife_"+FLAGS.output_dir_postfix), exist_ok=True)  # Create the output folder, if it doesn't already exist
    cfg.OUTPUT_DIR = os.path.join(Path(config_folder).parents[1], "output_vitrolife_"+FLAGS.output_dir_postfix) # Get second parent directory to config_folder, i.e. MaskFormer folder and create an output directory
    cfg.SOLVER.BASE_LR = FLAGS.learning_rate if "LEARNING_RATE" in key_list else 1e-3       # Starting learning rate
    cfg.SOLVER.IMS_PER_BATCH = FLAGS.batch_size if "BATCH_SIZE" in key_list else 1          # Batch size used when training => batch_size pr GPU = batch_size // num_gpus
    cfg.SOLVER.MAX_ITER = FLAGS.max_iter if "MAX_ITER" in key_list else int(2e4)            # Maximum number of iterations to train for
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"                                      # Default learning rate scheduler
    cfg.SOLVER.NESTEROV = True                                                              # Whether or not the learning algorithm will use Nesterow momentum
    cfg.SOLVER.WEIGHT_DECAY = float(2e-5)                                                   # A small lambda value for the weight decay
    cfg.SOLVER.STEPS = [int(x+1)*100 for x in range(5)]                                     # The warm up steps for the learning rate scheduler
    cfg.SOLVER.CHECKPOINT_PERIOD = MetadataCatalog["vitrolife_dataset_train"].num_files_in_dataset  # Save a new model checkpoint after each epoch, i.e. after everytime the entire trainining set has been seen by the model
    cfg.TEST.EVAL_PERIOD = MetadataCatalog["vitrolife_dataset_train"].num_files_in_dataset  # Evaluation after each epoch. Thus in the logs it can be seen which iteration was "best" and then that checkpoint can be loaded later
    cfg.TEST.AUG.FLIP = False 
    cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED = False

    # Write the new config as a .yaml file - it already does, in the output dir...
    with open(os.path.join(cfg.OUTPUT_DIR, "vitrolife_config_initial.yaml"), "w") as f:
        f.write(cfg.dump())
    f.close()
        
    # Return the custom configuration
    return cfg

