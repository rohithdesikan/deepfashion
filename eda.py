#%%
# Import base packages
import numpy as np
import pandas as pd
import json
import time
import os 

# Import visualization packages
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, transform

# Import ML libraries
import torch
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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


# %%
# Iterate through the image list to find the largest one

def find_largest_image(path_images, image_fns):
    num_images = len(image_list)
    sizes = np.zeros((num_images, 2))

    for i, image in enumerate(image_fns):

        # Set up image path and open the single image
        image_path = os.path.join(path_images, image)
        image = Image.open(image_path)

        # Output image size to an array
        sizes[i, 0], sizes[i ,1] = image.size

    return sizes

# %%
# Run the find largest image function to find largest image
# s = time.time()
# sizes = find_largest_image(path_train_images, image_list)
# print(sizes.max())
# e = time.time()
# print("Time taken to iterate through all images: ", e - s)

# Max Size: Width- 1320, Height- 1835
# Min Size: Width- 68, Height- 71

# %%
target_height, target_width = 1200, 600

# %%
# Rescale images and generate the list of images, targets and bounding boxes

# Initialize an empty list of the length of images and a list of targets which is a list of dictionaries as expected by PyTorch
images = [None] * len(image_list_small)
targets = [{}] * len(image_list_small)

for i, (image, anno) in enumerate(zip(image_list_small, annos_list_small)):

    # Output image size to a numpy array
    path_image = os.path.join(path_train_images, image)
    img = Image.open(path_image)
    image_orig = np.array(img)

    # Get the original size of the array
    h_orig, w_orig = image_orig.shape[:2]

    # Transform the array to a new target size specified above
    trfm = transforms.Resize((target_height, target_width))
    image_scaled = trfm(img)

    # Convert the scaled PIl image to a np array
    image_arr = np.array(image_scaled)

    # Move the numpy axes to the right order as accepted by Pytorch
    image_channels = np.moveaxis(image_arr, 2, 0)
    image_ordered = np.moveaxis(image_channels, 2, 1)


    # Load the annotations as a json
    path_anno = os.path.join(path_train_annos, anno)
    with open(path_anno,'r') as f:
        data = json.load(f)
    

    # Number of targets/labels/bounding boxes and initialize a final targets dictionary for PyTorch input
    num_targets = len([k for k in list(data.keys()) if 'item' in k])

    # Initialize 2 lists of labels and bboxes per image. There can obviously be multiple
    labels_per_image = [None] * num_targets
    boxes_scaled_per_image = [None] * num_targets

    # Iterate through each annotation to find the # of labels and associated bounding boxes in the image
    for j in range(1,num_targets+1):
        
        # Obtain the image label
        labels_per_image[j-1] = data[f'item{j}']['category_id']

        # Get the bounding box and normalize it by the width and height calculated above
        x0, y0, x1, y1 = data[f'item{j}']['bounding_box']

        # Calculate new bounding boxes with the target size
        x0_new = x0 * target_width / w_orig
        y0_new = y0 * target_height / h_orig
        x1_new = x1 * target_width / w_orig
        y1_new = y1 * target_height / h_orig

        # Put the bounding box information by label into a specific list
        boxes_scaled_per_image[j-1] = [x0_new, y0_new, x1_new, y1_new]
    

    # Convert the list of labels into a tensor
    labels_tensor = torch.IntTensor(labels_per_image)

    # Convert the list of lists of bounding boxes to a torch float tensor
    boxes_tensor = torch.FloatTensor(boxes_scaled_per_image)

    # Convert the image array to a pytorch tensor
    image_tensor = torch.from_numpy(image_ordered)

    # print("Iteration # ", i)

    # Put the image tensor into that element of the master list of images
    images[i] = image_tensor

    # Put the targets as a dictionary into the ith element of the master list of targets which includes labels and bounding boxes
    target_dict_per_image = {'labels': labels_tensor, 'boxes': boxes_tensor}
    targets[i] = target_dict_per_image

    # print("Targets: ", targets[i])



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

num_classes = 14  # 13 classes + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# %%
# Generate training data
images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
labels = torch.randint(1, 91, (4, 11))
images = list(image for image in images)
targets = []
for i in range(len(images)):
    d = {}
    d['boxes'] = boxes[i]
    d['labels'] = labels[i]
    targets.append(d)

# output = model(images, targets)

# %%
# For inference
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)

# %%
# arr2 = np.moveaxis(arr, 2, 0)
# arr3 = np.moveaxis(arr2, 2, 1)
# arr4 = np.reshape(arr3, (1, 3, 1034, 688))
# arr5 = [torch.from_numpy(arr3)]

# %%
output = model(images, targets)
# model.eval()
# predictions = model(images)

# %%
# Steps to clean and preprocess the images and data

# Initialize a list of dictionaries for bounding boxes and labels

    # Change axes order from [W, H, C] to [C, H, W] 

    # Collect it into the numpy array of standardized size (Call the custom resize function to resize both the image and the associated bounding boxes)

    # Collect the label (category_id) and bounding boxes in a list of dictionaries according to: https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
    
    # Easier to collect in a list of dictionaries that gets processed at every image as every image can have a variable number of teh 13 detected clothing items

# Build and train the model on only 1000 images. Small enough that it would work quickly. 

# Make an inference on ~100 processed images.


