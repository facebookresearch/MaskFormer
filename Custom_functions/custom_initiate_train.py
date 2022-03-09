# Add the MaskFormer directory to PATH
import os                                                                   # Used to navigate the folder structure in the current os
import sys                                                                  # Used to control the PATH variable
MaskFormer_dir = os.path.join("/mnt", "c", "Users", "Nico-", "Documents", "Python_Projects", "MaskFormer")                                                              # Home WSL
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("C:\\", MaskFormer_dir.split(os.path.sep, 1)[1])                                                    # Home windows computer
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Repositories", "MaskFormer")  # Larac server
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("/mnt", "home_shared", MaskFormer_dir.split(os.path.sep, 2)[2])                                     # Balder server
assert os.path.isdir(MaskFormer_dir), "The MaskFormer directory doesn't exist in the chosen location"
sys.path.append(MaskFormer_dir)                                             # Add MaskFormer directory to PATH
sys.path.append(os.path.join(MaskFormer_dir, "Custom_functions"))           # Add Custom_functions directory to PATH
sys.path.append(os.path.join(MaskFormer_dir, "tools"))                      # Add the tools directory to PATH

# Add the environmental variable DETECTRON2_DATASETS
dataset_dir = os.path.join("/mnt", "c", "Users", "Nico-", "OneDrive - Aarhus Universitet", "Biomedicinsk Teknologi", "5. semester", "Speciale", "Datasets")             # Home WSL
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("C:\\", dataset_dir.split(os.path.sep,1)[1])                                                              # Home windows computer
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Datasets")                          # Larac server
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/mnt", "home_shared", dataset_dir.split(os.path.sep, 2)[2])                                              # Balder server
assert os.path.isdir(dataset_dir), "The dataset directory doesn't exist in the chosen location"
os.environ["DETECTRON2_DATASETS"] = dataset_dir

# Import important libraries
import argparse                                                             # Used to parse input arguments through command line
from datetime import datetime                                               # Used to get the current date and time when starting the process
from create_custom_config import createVitrolifeConfiguration               # Function to create the custom configuration used for the training with Vitrolife dataset
from detectron2.engine import default_argument_parser                       # Default argument_parser object
from custom_train_func import launch_custom_training                        # Function to launch the training with custom dataset
from visualize_vitrolife_batch import visualize_the_images                  # Import the function used for visualizing the image batch
from GPU_memory_ranked_assigning import assign_free_gpus                    # Function to assign the running process to a specified number of GPUs ranked by memory availability
from show_learning_curves import show_history                               # Function used to plot the learning curves for the given training


# Function to rename the automatically created "inference" directory in the OUTPUT_DIR from "inference" to "validation" before performing actual inference with the test set
def rename_output_inference_folder(config):                                 # Define a function that will only take the config as input
    source_folder = os.path.join(config.OUTPUT_DIR, "inference")            # The source folder is the current inference (i.e. validation) directory
    dest_folder = os.path.join(config.OUTPUT_DIR, "validation")             # The destination folder is in the same parent-directory where inference is changed with validation
    os.rename(source_folder, dest_folder)                                   # Perform the renaming of the folder

# Define a function to convert string values into booleans
def str2bool(v):
    if isinstance(v, bool): return v                                        # If the input argument is already boolean, the given input is returned as-is
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True             # If any signs of the user saying yes is present, the boolean value True is returned
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False          # If any signs of the user saying no is present, the boolean value False is returned
    else: raise argparse.ArgumentTypeError('Boolean value expected.')       # If the user gave another input an error is raised


# Define the main function used to send input arguments. Just return the FLAGS arguments as a namespace variable
def main(FLAGS):
    assign_free_gpus(max_gpus=1)                                            # Working with the Vitrolife dataset can only be done using a single GPU for some weird reason...
    cfg = createVitrolifeConfiguration(FLAGS=FLAGS)                         # Register the vitrolife datasets and create the custom configuration

    # Visualize some random images
    fig, filename_dict, cfg = visualize_the_images(config=cfg, FLAGS=FLAGS) # Visualize some segmentations on validation images before training
    if FLAGS.display_images: fig.show()

    # Train the model
    launch_custom_training(args=FLAGS, config=cfg)                          # Launch the training loop

    # Visualize the same images, now with a trained model
    fig, filename_dict, cfg = visualize_the_images(config=cfg, FLAGS=FLAGS, filename_dict=filename_dict)    # Visualize the same images, from either train/val split after training
    if FLAGS.display_images: fig.show()

    # Evaluation on the test dataset
    if FLAGS.debugging == False:                                            # Inference will only be performed if we are not debugging the model
        rename_output_inference_folder(config=cfg)                          # Rename the "inference" folder in OUTPUT_DIR to "validation" before doing inference
        FLAGS.eval_only = True                                              # Letting the model know we will only perform evaluation here
        cfg.DATASETS.TEST = ("vitrolife_dataset_test",)                     # The inference will be done on the test dataset
        launch_custom_training(args=FLAGS, config=cfg)                      # Launch the training (i.e. validation) loop
        fig,_,_=visualize_the_images(config=cfg, FLAGS=FLAGS)               # Visualizing some new segmented test images
    
    # Display learning curves
    fig = show_history(config=cfg, save_folder=cfg.OUTPUT_DIR)              # Create and save learning curves


# Running the main function
if __name__ == "__main__":
    # Create the input arguments with possible values
    parser = default_argument_parser()
    start_time = datetime.now().strftime("%H_%M_%d%b%Y").upper()
    parser.add_argument("--dataset_name", type=str, default="Vitrolife", help="Which datasets to train on. Choose between [ADE20K, Vitrolife]. Default: Vitrolife")
    parser.add_argument("--output_dir_postfix", type=str, default=start_time, help="Filename extension to add to the output directory of the current process. Default: now: 'HH_MM_DDMMMYYYY'")
    parser.add_argument("--Model_weights", type=str, default="", help="Path to the checkpoint [.pth, .pkl], to initialize model weights. If empty, initialize model weights randomly. Default: ''")
    parser.add_argument("--Num_workers", type=int, default=1, help="Number of workers to use for training the model. Default: 1")
    parser.add_argument("--max_iter", type=int, default=int(1.5e2), help="Maximum number of iterations to train the model for. Default: 100")
    parser.add_argument("--Img_size_min", type=int, default=500, help="The length of the smallest size of the training images. Default: 500")
    parser.add_argument("--Img_size_max", type=int, default=500, help="The length of the largest size of the training images. Default: 500")
    parser.add_argument("--Resnet_Depth", type=int, default=50, help="The depth of the feature extracting ResNet backbone. Possible values: [18,34,50,101] Default: 50")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size used for training the model. Default: 1")
    parser.add_argument("--num_images", type=int, default=5, help="The number of images to display. Only relevant if --display_images is true. Default: 5")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="The initial learning rate used for training the model. Default 1e-4")
    parser.add_argument("--Crop_Enabled", type=str2bool, default=False, help="Whether or not cropping is allowed on the images. Default: False")
    parser.add_argument("--display_images", type=str2bool, default=True, help="Whether or not some random sample images are displayed before training starts. Default: False")
    parser.add_argument("--debugging", type=str2bool, default=False, help="Whether or not we are debugging the script. Default: False")
    # Parse the arguments into a Namespace variable
    FLAGS = parser.parse_args()
    FLAGS = main(FLAGS)

