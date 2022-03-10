import os                                                                                   # Used to navigate around the current os and folder structure
import torch                                                                                # torch is implemented to check if a GPU is available
import numpy as np
from sys import path as sys_PATH                                                            # Import the PATH variable
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
    # Locate the folder containing other configurations
    MaskFormer_dir = [x for x in sys_PATH if x.endswith("MaskFormer")][0]
    checkpoint_dir = os.path.join(MaskFormer_dir, "maskformer_model_checkpoints")
    config_folder = os.path.join(MaskFormer_dir, "configs", "ade20k-150")

    # Get all keys from the FLAGS input argument
    key_list = accumulate_keys(vars(FLAGS))

    # Alter the configuration and make it as wanted. Not all keys will be used, e.g. Panoptic keys are unused for now
    # cfg = CN(CN.load_yaml_with_base(os.path.join(config_folder, "maskformer_R50_bs16_160k.yaml"), allow_unsafe=True))
    cfg = get_cfg()                                                                         # Get the default configuration from detectron2.
    add_deeplab_config(cfg)                                                                 # Add some deeplab (i.e. sem_seg) config values
    add_mask_former_config(cfg)                                                             # Add some default values used for semantic segmentation to the config and choose datasetmapper
    if FLAGS.use_transformer_backbone==True and FLAGS.use_per_pixel_baseline==False:        # If the user chose the transformer backbone ...
        cfg.merge_from_file(os.path.join(config_folder, "swin", "maskformer_swin_small_bs16_160k.yaml"))    # ... merge with the config for the small swin transformer
        if FLAGS.use_checkpoint==True:                                                      # If the user choose to start training from a earlier checkpoint ...
            cfg.MODEL.WEIGHTS = os.path.join(checkpoint_dir, "maskformer_swin_small_checkpoint.pkl")    # Load the swin_transformer checkpoint
    elif FLAGS.use_transformer_backbone==False and FLAGS.per_pixel_baseline==False:         # Else-if the user will use regular ResNet backbone ...
        cfg.merge_from_file(os.path.join(config_folder, "maskformer_R50_bs16_160k.yaml"))   # ... merge with the ResNet_maskformer config
        cfg.MODEL.RESNETS.DEPTH = FLAGS.resnet_depth                                        # Assign the depth of the ResNet backbone feature extracting model
        if FLAGS.use_checkpoint==True:                                                      # If the user choose to start training from a earlier checkpoint ...
            cfg.MODEL.WEIGHTS = os.path.join(checkpoint_dir, "maskformer_resnet_backbone_checkpoint.pkl")   # Load the resnet_backbone checkpoint
    elif FLAGS.use_per_pixel_baseline==True:                                                # Otherwise, then the user chose to work with the per_pixel_calculating baselines, so ...
        cfg.merge_from_file(os.path.join(config_folder, "per_pixel_baseline_R50_bs16_160k.yaml")) # ... merge with the per_pixel_baseline config
        if FLAGS.use_checkpoint==True:                                                      # If the user choose to start training from a earlier checkpoint ...
            cfg.MODEL.WEIGHTS = os.path.join(checkpoint_dir, "maskformer_per_pixel_baseline_checkpoint.pkl")    # Load the per_pixel classification checkpoint
    cfg.merge_from_file(os.path.join(config_folder, "Base-ADE20K-150.yaml"))                # Merge with the base config for ade20K dataset. This is the config selecting that we use the ADE20K dataset
    cfg.SOLVER.BASE_LR = FLAGS.learning_rate if "LEARNING_RATE" in key_list else 1e-3       # Starting learning rate
    cfg.SOLVER.IMS_PER_BATCH = FLAGS.batch_size if "BATCH_SIZE" in key_list else 1          # Batch size used when training => batch_size pr GPU = batch_size // num_gpus
    cfg.SOLVER.MAX_ITER = FLAGS.max_iter if "MAX_ITER" in key_list else int(2e4)            # Maximum number of iterations to train for
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"                                      # Default learning rate scheduler
    cfg.SOLVER.NESTEROV = True                                                              # Whether or not the learning algorithm will use Nesterow momentum
    cfg.SOLVER.WEIGHT_DECAY = float(2e-5)                                                   # A small lambda value for the weight decay
    cfg.TEST.AUG.FLIP = False                                                               # No random flipping or augmentation used for inference
    cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED = False                                          # Disable the panoptic head during inference
    cfg.DATALOADER.NUM_WORKERS = FLAGS.num_workers                                          # Set the number of workers to only 2
    cfg.INPUT.CROP.ENABLED =  FLAGS.crop_enabled                                            # We will not allow any cropping of the input images
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'                       # Assign the device on which the model should run
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 5                                                   # Set the weight for the dice loss
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20                                                  # Set the weight for the mask predictive loss
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5                                           # Assign the threshold used for the model
    cfg.OUTPUT_DIR = os.path.join(MaskFormer_dir, "output_"+FLAGS.output_dir_postfix)       # Get MaskFormer directory and name the output directory
    config_name = "config_initial.yaml"                                                     # Initial name for the configuration that will be saved in the cfg.OUTPUT_DIR
    if "vitrolife" in FLAGS.dataset_name.lower():                                           # ... and the default value of vitrolife is chosen ...
        cfg["DATASETS"]["TRAIN"] = ("vitrolife_dataset_train",)                             # ... define the training dataset by using the config as a dictionary
        cfg.DATASETS.TEST = ("vitrolife_dataset_val",)                                      # ... define the validation dataset by using the config as a CfgNode 
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(MetadataCatalog[cfg.DATASETS.TEST[0]].stuff_classes)   # Assign the number of classes for the model to segment
        cfg.INPUT.FORMAT = "BGR"                                                            # The input format is set to be BGR, like the visualization method
        cfg.INPUT.MIN_SIZE_TRAIN = FLAGS.img_size_min                                       # The minimum size length for one side of the training images
        cfg.INPUT.MAX_SIZE_TRAIN = FLAGS.img_size_max                                       # The maximum size length for one side of the training images
        cfg.INPUT.MIN_SIZE_TEST = FLAGS.img_size_min                                        # The minimum size length for one side of the validation images
        cfg.INPUT.MAX_SIZE_TEST = FLAGS.img_size_max                                        # The maximum size length for one side of the validation images
        cfg.MODEL.PIXEL_MEAN = [100.15, 102.03, 103.89]                                     # Write the correct image mean value for the entire vitrolife dataset
        cfg.MODEL.PIXEL_STD = [57.32, 59.69, 61.93]                                         # Write the correct image standard deviation value for the entire vitrolife dataset
        cfg.SOLVER.CHECKPOINT_PERIOD = MetadataCatalog[cfg.DATASETS.TRAIN[0]].num_files_in_dataset  # Save a new model checkpoint after each epoch, i.e. after everytime the entire trainining set has been seen by the model
        cfg.TEST.EVAL_PERIOD = MetadataCatalog[cfg.DATASETS.TEST[0]].num_files_in_dataset   # Evaluation after each epoch. Thus in the logs it can be seen which iteration was "best" and then that checkpoint can be loaded later
        cfg.SOLVER.STEPS = np.subtract([int(x+1)*np.min([500, cfg.SOLVER.MAX_ITER]) for x in range(500)],1).tolist()    # The iterations where the learning rate will be lowered with a factor of "gamma"
        cfg.SOLVER.GAMMA = 0.25                                                             # After every "step" iterations the learning rate will be updated, as new_lr = old_lr*gamma
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace("output_", "output_vitrolife_")             # Insert the 'vitrolife' to the output directory, if using the vitrolife dataset
        config_name = "vitrolife_" + config_name                                            # Prepend the config name with "vitrolife"
    if FLAGS.debugging==True:                                                               # If we are debugging the model ...
        cfg.SOLVER.CHECKPOINT_PERIOD = int(np.subtract(cfg.SOLVER.MAX_ITER, 1))             # ... a checkpoint will only be saved after the final iteration
        cfg.TEST.EVAL_PERIOD = int(np.subtract(cfg.SOLVER.MAX_ITER, 1))                     # ... inference will only happen after the final iteration
        cfg.DATASETS.TEST = cfg.DATASETS.TRAIN                                              # ... and inference will happen on the training set

    # Write the new config as a .yaml file - it already does, in the output dir...
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)                                              # Create the output folder, if it doesn't already exist
    with open(os.path.join(cfg.OUTPUT_DIR, config_name), "w") as f:                         # Open a object instance with the config file
        f.write(cfg.dump())                                                                 # Dump the configuration to a file named config_name in cfg.OUTPUT_DIR
    f.close()
    
    # Return the custom configuration
    return cfg

