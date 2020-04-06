#%%
# Import base packages
import numpy as np
import pandas as pd
import json
import time
import os 

# Import 
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt

# Import ML libraries
import torch
from torch.autograd import Variable
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# %%
# Set the file paths
curr_dir = os.getcwd()
print(curr_dir)
path_train = os.path.join(curr_dir,'datasets','train')
path_val = os.path.join(curr_dir,'datasets','validation')
path_train_images = os.path.join(path_train, 'image')
path_train_annos = os.path.join(path_train, 'annos')

# %%
image_list = os.listdir(path_train_images)
annos_list = os.listdir(path_train_annos)
num_images = len(image_list)

# %%
sizes = np.zeros((num_images, 2))
labels = np.zeros((num_images, 5))
image_arrs = np.zeros((num_images))

for i, (image, anno) in enumerate(zip(image_list, annos_list)):

    # Output image size to an array
    image = Image.open(image)
    sizes[i, 0], sizes[i ,1] = image.size


    # Load an image
    with open(anno,'r') as f:
        data = json.load(f)

    # Iterate through each annotation to find the # of labels in the image
    num_objs = len([k for k in list(data.keys()) if 'item' in k])

    # Iterate through each item to find the category id
    for obj in range(1,num_objs+1):
        labels[i, 0] = data[f'item{obj}']['category_id']

        # Get the bounding box and normalize it by the width and height calculated above
        x0_full, y0_full , x1_full , y1_full = data[f'item{obj}']['bounding_box']
        x0, y0 ,x1, y1 = x0_full/sizes[i,0], y0_full/sizes[i,1] , x1_full/sizes[i,0] , y1_full/sizes[i,1]


# %%
# Set up a single image to view
trial_image = '022243'

# Create image path and read in image
path_image = os.path.join(path_train_images, f'{trial_image}.jpg')
image = Image.open(path_image)
width, height = image.size
arr = np.array(image).reshape((width, height, 3))
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

# %%
# Find the maximum number of items aka bounding boxes in a single image

#region
# s = time.time()
# annos_files = os.listdir(os.path.join(path_train,'annos'))
# print(len(annos_files))

# max_num_items = 0
# max_files = []

# for f in annos_files:
#     annos_path = os.path.join(path_train,'annos',f)
    
#     with open(annos_path, 'r') as annos_file:
#         data = json.load(annos_file)
    
#     num_items = len([k for k in list(data.keys()) if 'item' in k])
#     max_num_items = max(num_items,max_num_items)
    
#     if num_items==8:
#         max_files.append(f)
#     else:
#         pass

# # Happens to be 8 at Img#095484
# print("Max # Items: ", max_num_items)
# e = time.time()
# print(e-s)

#endregion

# %%
# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 13  # 13 classes + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# %%
images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
labels = torch.randint(1, 91, (4, 11))
images = list(image for image in images)
targets = []
for i in range(len(images)):
    d = {}
    d['boxes'] = boxes[i]
    d['labels'] = labels[i]
    targets.append(d)
output = model(images, targets)
# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)

# %%
# Steps to clean and preprocess the images and data

# Iterate through the images:
    # Find the largest image by dimensions in the dataset

# Initialize a list of dictionaries for bounding boxes and labels
# Initialize a numpy array of the largest image size 

# Iterate again over the images and annotations and at every iteration: 
    # Zero pad all the pixels (with black) on the right and bottom from the maximum size. This way, the bounding box remains the same so that all images are the same size. Whatever size this comes up with, try putting that into the R-CNN model.

    # Change axes order from [W, H, C] to [C, H, W] 

    # Collect it into the numpy array

    # Collect the label (category_id) and bounding boxes in a list of dictionaries according to: https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py


# Make an inference on just 1 processed image. 


