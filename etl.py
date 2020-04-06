# %%
# Import base packages
import numpy as np
import pandas as pd
import os
import json
import time
import yaml

# Import viz packages
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns

# Import ML packages
import cv2
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter



# %%
# Read in YAML file
curr_dir = os.getcwd()

with open('config.yml','r') as ymlfile:
    inputs = yaml.load(ymlfile, Loader=yaml.FullLoader)
image_files_path = inputs['data']['X_train']
image_files = os.listdir(inputs['data']['X_train'])

# For testing
num_images = 191961
single_image_file = '000111.jpg'
single_image_path = os.path.join(image_files_path, single_image_file)

# %%
# Create X_train matrix (191961 * pixels_x * pixels_y * 3)
# s = time.time()

# x_size = np.zeros([num_images,1])
# y_size = np.zeros([num_images,1])
# c_size = np.zeros([num_images,1])

# for ind, image_file in enumerate(image_files):
#     image_fpath = os.path.join(inputs['data']['X_train'],image_file)
#     image = cv2.imread(image_fpath)
#     x_size[ind-1,:] = image.shape[0]
#     y_size[ind-1,:] = image.shape[1]
#     c_size[ind-1:] = image.shape[2]

# e = time.time()
# print("Elapsed_Time for image size information: ", e-s)

# %%
# def load_images(image_files):
#     for image in image_files:
image_raw = load_img(single_image_path)
plt.imshow(image_raw)

image = load_img(single_image_path, target_size=(682,682))
arr = img_to_array(image)
arr = arr.astype('float32')
arr /= 255.0
image_resized = array_to_img(arr)
# plt.imshow(image_resized)


# %%
