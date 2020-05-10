# %%
# Import base packages
import os 
import json
import argparse

from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import skimage

# %%
# Set the file paths
curr_dir = os.getcwd()

path_train = os.path.join(curr_dir,'datasets','train')
path_val = os.path.join(curr_dir,'datasets','validation')
path_train_images = os.path.join(path_train, 'image')
path_train_annos = os.path.join(path_train, 'annos')

path_train_processed = os.path.join(curr_dir,'datasets','processed_train')
path_val_processed = os.path.join(curr_dir,'datasets','processed_validation')
path_train_images_processed = os.path.join(path_train_processed, 'image')
path_train_annos_processed = os.path.join(path_train_processed, 'annos')

image_list = os.listdir(path_train_images)
annos_list = os.listdir(path_train_annos)
num_images = len(image_list)

image_list_small = image_list[:100]
annos_list_small = annos_list[:100]

# %%
# Create a protobuf label map
# def resize_images


# %%
