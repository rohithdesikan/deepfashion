# %%
# Import base packages
import numpy as np
import pandas as pd
import os 
import json
import time
import argparse
import imutils
import random
from PIL import Image, ImageDraw

import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import keras
import tensorflow
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

import mrcnn
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize

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
num_images_small = len(image_list_small)

# %%
# Set up a single image to view
# trial_image = '095484'
trial_image = '022373'

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
xl = []
yl = []

for i in range(1,num_items+1):
    label = data[f'item{i}']['category_id']
    x0,y0,x1,y1 = data[f'item{i}']['bounding_box']

    lx = data[f'item{i}']['landmarks'][::3]
    ly = data[f'item{i}']['landmarks'][1::3]
    ls = [(x, y) for (x, y) in zip(lx, ly)]

    draw = ImageDraw.Draw(image)
    draw.polygon(ls, outline='red')

    # draw = ImageDraw.Draw(image)
    # draw.rectangle(tuple(data[f'item{i}']['bounding_box']), outline='red')

plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(image)

# %%
