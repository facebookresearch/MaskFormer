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
from custom_setup_func import rename_output_inference_folder, setup_func, zip_output    # Functions to rename output dir and assign to GPU, register vitrolife dataset, create config and zip output_dir
from custom_train_func import launch_custom_training                        # Function to launch the training with custom dataset
from visualize_vitrolife_batch import visualize_the_images                  # Import the function used for visualizing the image batch
from show_learning_curves import show_history                               # Function used to plot the learning curves for the given training

# Get the FLAGS and config variables
FLAGS, cfg = setup_func()

# Visualize some random images
fig_list_before, data_batches, cfg, FLAGS = visualize_the_images(config=cfg, FLAGS=FLAGS)   # Visualize some segmentations on random images before training

# Train the model
launch_custom_training(args=FLAGS, config=cfg)                              # Launch the training loop

# Visualize the same images, now with a trained model
fig_list_after, data_batches, cfg, FLAGS = visualize_the_images(            # Visualize the same images ...
    config=cfg,FLAGS=FLAGS, data_batches=data_batches, model_has_trained=True)  # ... now after training

# Evaluation on the vitrolife test dataset. There is no ADE20K test dataset.
if FLAGS.debugging == False and "vitrolife" in FLAGS.dataset_name.lower():  # Inference will only be performed if we are not debugging the model
    rename_output_inference_folder(config=cfg)                              # Rename the "inference" folder in OUTPUT_DIR to "validation" before doing inference
    FLAGS.eval_only = True                                                  # Letting the model know we will only perform evaluation here
    cfg.DATASETS.TEST = ("vitrolife_dataset_test",)                         # The inference will be done on the test dataset
    launch_custom_training(args=FLAGS, config=cfg)                          # Launch the training (i.e. inference) loop

# Display learning curves
fig_learn_curves = show_history(config=cfg, FLAGS=FLAGS)                    # Create and save learning curves

# Zip the resulting output directory
zip_output(cfg)



