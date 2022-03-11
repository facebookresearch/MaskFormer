# Import libraries
import os                                                                   # Used to navigate the folder structure in the current os
import argparse                                                             # Used to parse input arguments through command line
from datetime import datetime                                               # Used to get the current date and time when starting the process
from shutil import make_archive                                             # Used to zip the directory of the output folder
from detectron2.data import DatasetCatalog, MetadataCatalog                 # Catalogs over registered datasets and metadatas for all datasets available in Detectron2
from detectron2.engine import default_argument_parser                       # Default argument_parser object
from GPU_memory_ranked_assigning import assign_free_gpus                    # Function to assign the script to a given number of GPUs
from register_vitrolife_dataset import register_vitrolife_data_and_metadata_func    # Register the vitrolife dataset and metadata in the Detectron2 dataset catalog
from create_custom_config import createVitrolifeConfiguration               # Function to create a configuration used for training 


# Function to rename the automatically created "inference" directory in the OUTPUT_DIR from "inference" to "validation" before performing actual inference with the test set
def rename_output_inference_folder(config):                                 # Define a function that will only take the config as input
    source_folder = os.path.join(config.OUTPUT_DIR, "inference")            # The source folder is the current inference (i.e. validation) directory
    dest_folder = os.path.join(config.OUTPUT_DIR, "validation")             # The destination folder is in the same parent-directory where inference is changed with validation
    os.rename(source_folder, dest_folder)                                   # Perform the renaming of the folder

def zip_output(cfg):
    print("Zipping the output directory {:s} with {:.0f} files".format(os.path.basename(cfg.OUTPUT_DIR), len(os.listdir(cfg.OUTPUT_DIR))))
    make_archive(base_name=os.path.basename(cfg.OUTPUT_DIR), format="zip",  # Instantiate the zipping of the output directory  where the resulting zip file will ...
        root_dir=os.path.dirname(cfg.OUTPUT_DIR), base_dir=os.path.basename(cfg.OUTPUT_DIR))    # ... include the output folder (not just the files from the folder)

# Define a function to convert string values into booleans
def str2bool(v):
    if isinstance(v, bool): return v                                        # If the input argument is already boolean, the given input is returned as-is
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True             # If any signs of the user saying yes is present, the boolean value True is returned
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False          # If any signs of the user saying no is present, the boolean value False is returned
    else: raise argparse.ArgumentTypeError('Boolean value expected.')       # If the user gave another input an error is raised


# Alter the FLAGS input arguments
def changeFLAGS(FLAGS):
    if FLAGS.num_gpus != FLAGS.gpus_used: FLAGS.num_gpus = FLAGS.gpus_used  # As there are two input arguments where the number of GPUs can be assigned, the gpus_used argument is superior
    if "vitrolife" in FLAGS.dataset_name.lower() : FLAGS.num_gpus = 1       # Working with the Vitrolife dataset can only be done using a single GPU for some weird reason...
    if FLAGS.eval_only != FLAGS.inference_only: FLAGS.eval_only = FLAGS.inference_only  # As there are two inputs where "eval_only" can be set, inference_only is the superior
    return FLAGS

# Define the main function used to send input arguments. Just return the FLAGS arguments as a namespace variable
def main(FLAGS):
    return FLAGS

# Running the parser function. By doing it like this the FLAGS will get out of the main function
parser = default_argument_parser()
start_time = datetime.now().strftime("%H_%M_%d%b%Y").upper()
parser.add_argument("--dataset_name", type=str, default="vitrolife", help="Which datasets to train on. Choose between [ADE20K, Vitrolife]. Default: Vitrolife")
parser.add_argument("--output_dir_postfix", type=str, default=start_time, help="Filename extension to add to the output directory of the current process. Default: now: 'HH_MM_DDMMMYYYY'")
parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use for training the model. Default: 1")
parser.add_argument("--max_iter", type=int, default=int(3e1), help="Maximum number of iterations to train the model for. Default: 100")
parser.add_argument("--img_size_min", type=int, default=500, help="The length of the smallest size of the training images. Default: 500")
parser.add_argument("--img_size_max", type=int, default=500, help="The length of the largest size of the training images. Default: 500")
parser.add_argument("--resnet_depth", type=int, default=50, help="The depth of the feature extracting ResNet backbone. Possible values: [18,34,50,101] Default: 50")
parser.add_argument("--batch_size", type=int, default=1, help="The batch size used for training the model. Default: 1")
parser.add_argument("--num_images", type=int, default=5, help="The number of images to display. Only relevant if --display_images is true. Default: 5")
parser.add_argument("--gpus_used", type=int, default=1, help="The number of GPU's to use for training. Only applicable for training with ADE20K. This input argument deprecates the '--num-gpus' argument. Default: 1")
parser.add_argument("--learning_rate", type=float, default=5e-3, help="The initial learning rate used for training the model. Default 1e-4")
parser.add_argument("--crop_enabled", type=str2bool, default=False, help="Whether or not cropping is allowed on the images. Default: False")
parser.add_argument("--inference_only", type=str2bool, default=False, help="Whether or not training is skipped and only inference is run. This input argument deprecates the '--eval_only' argument. Default: False")
parser.add_argument("--display_images", type=str2bool, default=True, help="Whether or not some random sample images are displayed before training starts. Default: False")
parser.add_argument("--use_checkpoint", type=str2bool, default=True, help="Whether or not we are loading weights from a model checkpoint file before training. Only applicable when using ADE20K dataset. Default: False")
parser.add_argument("--use_transformer_backbone", type=str2bool, default=True, help="Whether or now we are using the extended swin_small_transformer backbone. Only applicable if '--use_per_pixel_baseline'=False. Default: False")
parser.add_argument("--use_per_pixel_baseline", type=str2bool, default=False, help="Whether or now we are using the per_pixel_calculating head. Alternative is the MaskFormer (or transformer) heads. Default: False")
parser.add_argument("--debugging", type=str2bool, default=True, help="Whether or not we are debugging the script. Default: False")
# Parse the arguments into a Namespace variable
FLAGS = parser.parse_args()
FLAGS = main(changeFLAGS(FLAGS))

# Setup functions
assign_free_gpus(max_gpus=FLAGS.num_gpus)                                   # Assigning the running script to the selected amount of GPU's with the largest memory available
register_vitrolife_data_and_metadata_func(debugging=FLAGS.debugging)        # Register the vitrolife dataset
cfg = createVitrolifeConfiguration(FLAGS=FLAGS)                             # Create the custom configuration used to e.g. build the model

# Return the values again
def setup_func(): return FLAGS, cfg