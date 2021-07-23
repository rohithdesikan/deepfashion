#%%
# Import base packages
import numpy as np
import json
import time
import os

# Import visualization packages
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import skimage
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# %%
# Set the file paths
curr_dir = os.getcwd()
path_sample = os.path.abspath(os.path.join(curr_dir, os.pardir, "data", "sample_data"))
path_raw = os.path.abspath(
    os.path.join(curr_dir, os.pardir, os.pardir, "data", "processed")
)

path_train = os.path.join(path_raw, "train")
path_val = os.path.join(path_raw, "validation")
path_train_images = os.path.join(path_train, "image")
path_train_annos = os.path.join(path_train, "annos")
path_val_images = os.path.join(path_val, "image")
path_val_annos = os.path.join(path_val, "annos")
path_sample_images = os.path.join(path_sample, "image")
path_sample_annos = os.path.join(path_sample, "annos")

train_image_list = os.listdir(path_train_images)
train_annos_list = os.listdir(path_train_annos)

image_list_small = path_sample_images[:100]
annos_list_small = path_sample_annos[:100]

image_list_eval = path_sample_images[101:130]
annos_list_eval = path_sample_annos[101:130]

file_ids = [i.split(".")[0] for i in image_list_small]
path_train_annos = os.path.join(path_train, "annos")
path_val_images = os.path.join(path_val, "image")
path_val_annos = os.path.join(path_val, "annos")
path_sample_images = os.path.join(path_sample, "image")
path_sample_annos = os.path.join(path_sample, "annos")

image_list_sample = os.listdir(path_sample_images)
annos_list_sample = os.listdir(path_sample_annos)

image_list_train = image_list_sample[:100]
annos_list_train = annos_list_sample[:100]

image_list_eval = path_sample_images[101:130]
annos_list_eval = path_sample_annos[101:130]

file_ids_train = [i.split(".")[0] for i in image_list_train]
file_ids_eval = [i.split(".")[0] for i in image_list_eval]

# target_height, target_width = 500, 400

# %%
# Set up a single image to view
trial_image = "000022"

# Create image path and read in image
path_image = os.path.join(path_train_images, f"{trial_image}.jpg")
image = Image.open(path_image)
width, height = image.size
arr = np.array(image)
plt.imshow(image)

# Create annotations path and read it in
path_annos = os.path.join(path_train_annos, f"{trial_image}.json")
with open(path_annos, "r") as f:
    data = json.load(f)

num_items = len([k for k in list(data.keys()) if "item" in k])

for i in range(1, num_items + 1):
    label = data[f"item{i}"]["category_id"]
    x0, y0, x1, y1 = data[f"item{i}"]["bounding_box"]

    draw = ImageDraw.Draw(image)
    draw.rectangle([(x0, y0), (x1, y1)], outline="red")

plt.imshow(image)

# %%
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# For training
images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
labels = torch.randint(1, 91, (4, 11))
images = list(image for image in images)
targets = []
for i in range(len(images)):
    d = {}
    d["boxes"] = boxes[i]
    d["labels"] = labels[i]
    targets.append(d)

# %%
# 4 RGB images each of height=600 and width = 1200
print(
    "Images: ", len(images), type(images), type(images[0]), images[0].shape, images[0]
)  # 4 x 3 x 600 x 1200

# Each of the 4 images has 11 bounding boxes (x0, y0, x1, y1)
print(
    "Boxes: ", len(boxes), type(boxes), type(boxes[0]), boxes[0].shape, boxes[0]
)  # 4 x 11 x 4

# For each of the 4 images, each of the 11 bounding boxes has 1 of 91 labels
print(
    "Labels: ", len(labels), type(labels), type(labels[0]), labels[0], len(labels[0])
)  # 4 x 11

# List[Dict] where 1 dict corresponds to 1 image with keys: 'boxes' and 'labels'
print("Targets: ", len(targets), type(targets), type(targets[0]), targets[0])
