#%%
# Import base packages
import numpy as np
import pandas as pd
import os 
import json
import time
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO

# Import visualization packages
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Import ML packages
import skimage
import keras
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# %%
# Set the file paths
curr_dir = os.getcwd()
path_train = os.path.join(curr_dir,'datasets','train')
path_val = os.path.join(curr_dir,'datasets','validation')
path_train_images = os.path.join(path_train, 'image')
path_train_annos = os.path.join(path_train, 'annos')

image_list = os.listdir(path_train_images)
annos_list = os.listdir(path_train_annos)
num_images = len(image_list)

image_list_small = image_list[:100]
annos_list_small = annos_list[:100]

# %%
# Set up a single image to view
#region
trial_image = '022243'

# Create image path and read in image
path_image = os.path.join(path_train_images, f'{trial_image}.jpg')
image = Image.open(path_image)
width, height = image.size
arr = np.array(image)
plt.imshow(image)

# Create annotations path and read it in
path_annos = os.path.join(path_train_annos, f'{trial_image}.json')
with open(path_annos,'r') as f:
    data = json.load(f)

num_items = len([k for k in list(data.keys()) if 'item' in k])

for i in range(1,num_items+1):
    label = data[f'item{i}']['category_id']
    x0,y0,x1,y1 = data[f'item{i}']['bounding_box']

    draw = ImageDraw.Draw(image)
    draw.rectangle([(x0, y0), (x1, y1)], outline='red')

plt.imshow(image)
#endregion

# %%
# %%
target_height, target_width = 1200, 600

# Rescale images and generate the list of images, targets and bounding boxes
# Initialize an empty list of the length of images and a list of targets which is a list of dictionaries as expected by PyTorch
inputs = []

for i, (image, anno) in enumerate(zip(image_list_small, annos_list_small)):

    # Output image size to a numpy array
    path_image = os.path.join(path_train_images, image)
    img = Image.open(path_image)
    image_orig = np.array(img)

    # Get the original size of the array
    h_orig, w_orig = image_orig.shape[:2]

    # Transform the array to a new target size specified above
    image_scaled = skimage.transform.resize(image_orig, (target_height, target_width))

    # Convert the scaled PIl image to a np array
    image_arr = np.array(image_scaled)

    # # Move the numpy axes to the right order as accepted by Pytorch
    # image_channels = np.moveaxis(image_arr, 2, 0)
    # image_ordered = np.moveaxis(image_channels, 2, 1)


    # Load the annotations as a json
    path_anno = os.path.join(path_train_annos, anno)
    with open(path_anno,'r') as f:
        data = json.load(f)
    

    # Number of targets/labels/bounding boxes and initialize a final targets dictionary for PyTorch input
    num_targets = len([k for k in list(data.keys()) if 'item' in k])

    # Iterate through each annotation to find the # of labels and associated bounding boxes in the image
    for j in range(1,num_targets+1):
        
        # Obtain the image label
        label = data[f'item{j}']['category_id']

        # Get the bounding box and normalize it by the width and height calculated above
        x0, y0, x1, y1 = data[f'item{j}']['bounding_box']

        # Calculate new bounding boxes with the target size
        x0_new = x0 * target_width / w_orig
        y0_new = y0 * target_height / h_orig
        x1_new = x1 * target_width / w_orig
        y1_new = y1 * target_height / h_orig

        inputs.append([path_image, x0, x1, y0, y1, label])


# %%
data = pd.DataFrame(inputs)
data.to_csv('train.txt', header=None, index=None, sep = ',')

# %%
