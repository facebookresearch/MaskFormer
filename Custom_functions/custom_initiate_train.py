# Add the MaskFormer directory to PATH
import os                                                                   # Used to navigate the folder structure in the current os
import sys                                                                  # Used to control the PATH variable
MaskFormer_dir = os.path.join("C:\\", "Users", "Nico-", "Documents", "Python_Projects", "MaskFormer")                                                                       # Home windows computer
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("/mnt", "c", MaskFormer_dir.split(os.path.sep, 1)[1])                                                   # Home WSL
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Repositories", "MaskFormer")      # Larac server
if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("/mnt", "home_shared", MaskFormer_dir.split(os.path.sep, 1)[1])                                         # Balder server
assert os.path.isdir(MaskFormer_dir), "The MaskFormer directory doesn't exist in the chosen location"
sys.path.append(MaskFormer_dir)                                             # Add MaskFormer directory to PATH
sys.path.append(os.path.join(MaskFormer_dir, "Custom_functions"))           # Add Custom_functions directory to PATH
sys.path.append(os.path.join(MaskFormer_dir, "tools"))                      # Add the tools directory to PATH

# Add the environmental variable DETECTRON2_DATASETS
dataset_dir = os.path.join("C:\\", "Users", "Nico-", "OneDrive - Aarhus Universitet", "Biomedicinsk Teknologi", "5. semester", "Speciale", "Datasets")                      # Home windows computer
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/mnt", "c", dataset_dir.split(os.path.sep,1)[1])                                                             # Home WSL
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Datasets")                              # Larac server
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/mnt", "home_shared", dataset_dir.split(os.path.sep, 1)[1])                                                  # Balder server
assert os.path.isdir(dataset_dir), "The dataset directory doesn't exist in the chosen location"
os.environ["DETECTRON2_DATASETS"] = dataset_dir

# Import important libraries
import argparse                                                             # Used to parse input arguments through command line
from create_custom_config import createVitrolifeConfiguration               # Function to create the custom configuration used for the training with Vitrolife dataset
from detectron2.engine import default_argument_parser                       # Default argument_parser object
from custom_train_func import launch_custom_training                        # Function to launch the training with custom dataset
from visualize_vitrolife_batch import visualize_the_images                  # Import the function used for visualizing the image batch


# Define a function to convert string values into booleans
def str2bool(v):
    if isinstance(v, bool): return v                                        # If the input argument is already boolean, the given input is returned as-is
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True             # If any signs of the user saying yes is present, the boolean value True is returned
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False          # If any signs of the user saying no is present, the boolean value False is returned
    else: raise argparse.ArgumentTypeError('Boolean value expected.')       # If the user gave another input an error is raised


# Define the main function used to send input arguments. Just return the FLAGS arguments as a namespace variable
def main(FLAGS):
    # Register the vitrolife datasets and create the custom configuration
    cfg = createVitrolifeConfiguration(FLAGS=FLAGS)

    # Visualize some random images
    if FLAGS.display_images:
        fig = visualize_the_images(num_images=4, position=[0.55, 0.08, 0.40, 0.75])
        fig.savefig(os.path.join(cfg.OUTPUT_DIR, "Batched_samples.jpg"), bbox_inches="tight")   # Save the figure

    # Train the model
    launch_custom_training(args=FLAGS, config=cfg)


# Running the main function
if __name__ == "__main__":
    # Create the input arguments with possible values
    parser = default_argument_parser()
    parser.add_argument("--Num_workers", type=int, default=1, help="Number of workers to use for training the model. Default: 1")
    parser.add_argument("--max_iter", type=int, default=int(1e1), help="Maximum number of iterations to train the model for. Default: 10")
    parser.add_argument("--Img_size_min", type=int, default=500, help="The length of the smallest size of the training images. Default: 500")
    parser.add_argument("--Img_size_max", type=int, default=500, help="The length of the largest size of the training images. Default: 500")
    parser.add_argument("--Resnet_Depth", type=int, default=50, help="The depth of the feature extracting ResNet backbone. Possible values: [18,34,50,101] Default: 50")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size used for training the model. Default: 1")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="The initial learning rate used for training the model. Default 1e-4")
    parser.add_argument("--Crop_Enabled", type=str2bool, default=False, help="Whether or not cropping is allowed on the images. Default: False")
    parser.add_argument("--display_images", type=str2bool, default=False, help="Whether or not some random sample images are displayed before training starts. Default: False")
    parser.add_argument("--debugging", type=str2bool, default=False, help="Whether or not we are debugging the script. Default: False")
    # Parse the arguments into a Namespace variable
    FLAGS = parser.parse_args()
    FLAGS = main(FLAGS)

