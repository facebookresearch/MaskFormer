import os                                                                                   # Used to navigate around the current os and folder structure
from register_vitrolife_dataset import register_vitrolife_data_and_metadata_func            # Import function to register the vitrolife datasets in Detectron2 
from detectron2.data import MetadataCatalog                                                 # Catalog containing metadata for all datasets available in Detectron2
from detectron2.config import get_cfg                                                       # Function to get the default configuration from Detectron2
from detectron2.projects.deeplab import add_deeplab_config                                  # Used to merge the default config with the deeplab config before training
import torch                                                                                # torch is implemented to check if a GPU is available
from mask_former import add_mask_former_config                                              # Used to add the new configuration to the list of possible configurations
from pathlib import Path                                                                    # Used to get parent folders and determine output folder for training
import yaml                                                                                 # Used to read the yaml file
from detectron2.config import CfgNode as CN                                                 # Used to read the configs directly from the yaml file
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
def createVitrolifeConfiguration(FLAGS=Namespace()):
    # Register the vitrolife datasets
    register_vitrolife_data_and_metadata_func()
    assert any(["vitrolife" in x for x in list(MetadataCatalog)]), "Datasets have not been registered correctly"


    # Locate the folder containing other configurations
    MaskFormer_dir = os.path.join("C:\\", "Users", "Nico-", "Documents", "Python_Projects", "MaskFormer")
    if not os.path.isdir(MaskFormer_dir):
        MaskFormer_dir = os.path.join("/mnt", "c", "Users", "Nico-", "Documents", "Python_Projects", "MaskFormer")
    if not os.path.isdir(MaskFormer_dir):
        MaskFormer_dir = os.path.join("/mnt", "home_shared", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Repositories", "MaskFormer")
    config_folder = os.path.join(MaskFormer_dir, "configs", "ade20k-150")
    FLAGS = Namespace()

    # Get all keys from the FLAGS input argument
    key_list = accumulate_keys(vars(FLAGS))

    # Alter the configuration and make it as wanted. Not all keys will be used, e.g. Panoptic keys are unused for now
    # cfg = CN(CN.load_yaml_with_base(os.path.join(config_folder, "maskformer_R50_bs16_160k.yaml"), allow_unsafe=True))
    cfg = get_cfg()                                                                         # Get the default configuration from detectron2.
    add_deeplab_config(cfg)                                                                 # Add some deeplab (i.e. sem_seg) config values
    add_mask_former_config(cfg)                                                             # Add some default values used for semantic segmentation to the config and choose datasetmapper
    cfg.merge_from_file(os.path.join(config_folder, "maskformer_R50_bs16_160k.yaml"))       # Merge with the small maskformer config
    cfg.merge_from_file(os.path.join(config_folder, "Base-ADE20K-150.yaml"))                # Merge with the base config for ade20K dataset
    cfg["DATASETS"]["TRAIN"] = ("vitrolife_dataset_train",)                                 # Define the training dataset by using the config as a dictionary
    cfg.DATASETS.TEST = ("vitrolife_dataset_val",)                                          # Define the validation dataset by using the config as a CfgNode 
    # cfg["DATASETS"]["TRAIN"] = ("ade20k_sem_seg_train",)
    # cfg.DATASETS.TEST = ("ade20k_sem_seg_val",)
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
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5                                           # Assign the threshold used for the model
    os.makedirs(os.path.join(Path(config_folder).parents[1], "output_vitrolife"), exist_ok=True)    # Create the output folder, if it doesn't already exist
    cfg.OUTPUT_DIR = os.path.join(Path(config_folder).parents[1], "output_vitrolife")       # Get second parent directory to config_folder, i.e. MaskFormer folder and create an output directory
    cfg.SOLVER.BASE_LR = FLAGS.learning_rate if "LEARNING_RATE" in key_list else 1e-3       # Starting learning rate
    cfg.SOLVER.IMS_PER_BATCH = FLAGS.batch_size if "BATCH_SIZE" in key_list else 1          # Batch size used when training => batch_size pr GPU = batch_size // num_gpus
    cfg.SOLVER.MAX_ITER = FLAGS.max_iter if "MAX_ITER" in key_list else int(2e2)            # Maximum number of iterations to train for
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"                                      # Default learning rate scheduler
    cfg.SOLVER.NESTEROV = True                                                              # Whether or not the learning algorithm will use Nesterow momentum
    cfg.SOLVER.WEIGHT_DECAY = float(0)                                                      # Initially we will have no weight decay
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = False

    # Write the new config as a .yaml file - it already does, in the output dir...
    with open(os.path.join(cfg.OUTPUT_DIR, "vitrolife_config.yaml"), "w") as f:
        f.write(cfg.dump())
    f.close()
        
    # Return the custom configuration
    return cfg

