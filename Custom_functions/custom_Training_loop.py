# Add the MaskFormer and custom_functions folders to PATH
import os                                                                   # Used to navigate the folder structure in the current os
import sys                                                                  # Used to control the PATH variable
MaskFormer_dir = os.path.join("C:\\", "Users", "Nico-", "Documents", "Python_Projects", "MaskFormer")
if not os.path.isdir(MaskFormer_dir):
    MaskFormer_dir = os.path.join("/mnt", "c", "Users", "Nico-", "Documents", "Python_Projects", "MaskFormer")
if not os.path.isdir(MaskFormer_dir):
    MaskFormer_dir = os.path.join("/mnt", "home_shared", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Datasets", "MaskFormer")
sys.path.append(MaskFormer_dir)                                             # Add MaskFormer directory to PATH
sys.path.append(os.path.join(MaskFormer_dir, "Custom_functions"))           # Add Custom_functions directory to PATH
sys.path.append(os.path.join(MaskFormer_dir, "tools"))                      # Add the tools directory to PATH

dataset_dirs = os.path.join("C:\\", "Users", "Nico-", "OneDrive - Aarhus Universitet", "Biomedicinsk Teknologi", "5. semester", "Speciale", "Datasets")
os.path.environ["DETECTRON2_DATASETS"] = dataset_dirs

# Import important libraries
import argparse                                                             # Used to parse input arguments through command line
from create_custom_config import createVitrolifeConfiguration               # Function to create the custom configuration used for the training with Vitrolife dataset
from detectron2.engine import DefaultTrainer, default_argument_parser       # Default training loop and default argument_parser object
from detectron2.data import build_detection_train_loader, DatasetCatalog, DatasetMapper    # Create a dataloader 
from detectron2.modeling import build_model                                 # Used to build a model that can be used for anything, directly from a configuration file
from detectron2.solver.build import build_lr_scheduler, build_optimizer     # Functions to build lr_scheduler and optimizer
from detectron2.engine import default_setup, launch
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

    # Build the dataloaders, model, lr_scheduler and optimizer using the config.
    train_loader = build_detection_train_loader(dataset=DatasetCatalog.get("vitrolife_dataset_train"), mapper=DatasetMapper(cfg, is_train=True), total_batch_size=3)
    model = build_model(cfg)
    optimizer = build_optimizer(cfg=cfg, model=model)
    lr_scheduler = build_lr_scheduler(cfg=cfg, optimizer=optimizer)

    # Visualize some random images
    if FLAGS.display_images:
        fig = visualize_the_images(num_images=4, position=[0.55, 0.08, 0.40, 0.75])

    
    # Train the model
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    return trainer.train()



# Running the main function
if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--Num_workers", type=int, default=1, help="Number of workers to use for training the model")
    parser.add_argument("--max_iter", type=int, default=int(1e2), help="Maximum number of iterations to train the model for")
    parser.add_argument("--Img_size_min", type=int, default=500, help="The length of the smallest size of the training images")
    parser.add_argument("--Img_size_max", type=int, default=500, help="The length of the largest size of the training images")
    parser.add_argument("--Resnet_Depth", type=int, default=50, help="The depth of the feature extracting ResNet backbone")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size used for training the model")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="The initial learning rate used for training the model")
    parser.add_argument("--Crop_Enabled", type=str2bool, default=False, help="Whether or not cropping is allowed on the images")
    parser.add_argument("--display_images", type=str2bool, default=False, help="Whether or not some random sample images are displayed before training starts")
    FLAGS = main(parser.parse_args())

    # Launc the main function
    launch(
    main,
    FLAGS.num_gpus,
    num_machines=FLAGS.num_machines,
    machine_rank=FLAGS.machine_rank,
    dist_url=FLAGS.dist_url,
    args=(FLAGS,),
    )


