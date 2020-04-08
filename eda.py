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

image_list_small = image_list[:1000]
annos_list_small = annos_list[:1000]
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
# Use a custom rescale function for both image and bounding box
def rescale(image, bbox, output_size):

    h, w = image.shape[:2]
    new_h, new_w = output_size

    trfm = transforms.Resize((new_h, new_w))
    scaled_image = trfm(image)

    bboxes[0], bboxes[1], bboxes[2], bboxes[3] = x0 * new_w / w, y0 * new_h / h, x1 * new_w / w, y1* new_h / h

    return scaled_image, bboxes

# %%
# Run the find largest image function to find largest image
s = time.time()
sizes = find_largest_image(path_train_images, image_list)
print(sizes.max())
e = time.time()
print("Time taken to iterate through all images: ", e - s)

# Max Size: Width- 1320, Height- 1835
# Min Size: Width- 68, Height- 71

# %%
# sizes = np.zeros((num_images_small, 2))
labels = np.zeros((num_images_small, 1))
bboxes = np.zeros((num_images_small, 4))

target_height, target_width = 600, 1200
images_arr = np.zeros((num_images_small, 3, target_height, target_width))

for i, (image, anno) in enumerate(zip(image_list_small, annos_list_small)):

    # Output image size to an array
    image = Image.open(image)
    img_arr = np.array(image)
    img_arr_c = np.moveaxis(img_arr, 2, 0)
    img_arr_h = np.moveaxis(img_arr_c, 2, 1)

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

# num_classes = 13  # 13 classes + background

# # get number of input features for the classifier
# in_features = model.roi_heads.box_predictor.cls_score.in_features

# # replace the pre-trained head with a new one
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

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

# %%
output = model(images, targets)

# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)

# %%
arr2 = np.moveaxis(arr, 2, 0)
arr3 = np.moveaxis(arr2, 2, 1)
# arr4 = np.reshape(arr3, (1, 3, 1034, 688))
arr5 = [torch.from_numpy(arr3)]

model.eval()
predictions = model(arr5)

# %%
# Steps to clean and preprocess the images and data

# Initialize a list of dictionaries for bounding boxes and labels

    # Change axes order from [W, H, C] to [C, H, W] 

    # Collect it into the numpy array of standardized size (Call the custom resize function to resize both the image and the associated bounding boxes)

    # Collect the label (category_id) and bounding boxes in a list of dictionaries according to: https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
    
    # Easier to collect in a list of dictionaries that gets processed at every image as every image can have a variable number of teh 13 detected clothing items

# Build and train the model on only 1000 images. Small enough that it would work quickly. 

# Make an inference on ~100 processed images.


