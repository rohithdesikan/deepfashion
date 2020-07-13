# %%
# Import base packages
import numpy as np
import json
import os

from PIL import Image

# Import PyTorch packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# %%
# Set the file paths
curr_dir = os.getcwd()
path_sample = os.path.abspath(os.path.join(curr_dir, os.pardir, 'data', 'sample_data'))
path_raw = os.path.abspath(os.path.join(curr_dir, os.pardir, os.pardir, 'data', 'raw'))

path_train = os.path.join(path_raw, 'train')
path_val = os.path.join(path_raw, 'validation')
path_train_images = os.path.join(path_train, 'image')
path_train_annos = os.path.join(path_train, 'annos')
path_val_images = os.path.join(path_val, 'image')
path_val_annos = os.path.join(path_val, 'annos')
path_sample_images = os.path.join(path_sample, 'image')
path_sample_annos = os.path.join(path_sample, 'annos')

train_image_list = os.listdir(path_train_images)
train_annos_list = os.listdir(path_train_annos)

image_list_small = path_sample_images[:100]
annos_list_small = path_sample_annos[:100]

image_list_eval = path_sample_images[101:130]
annos_list_eval = path_sample_annos[101:130]

file_ids = [i.split('.')[0] for i in image_list_small]
path_train_targets = os.path.join(path_train, 'targets')
path_val_images = os.path.join(path_val, 'image')
path_val_targets = os.path.join(path_val, 'targets')
path_sample_images = os.path.join(path_sample, 'image')
path_sample_targets = os.path.join(path_sample, 'targets')

image_list_sample = os.listdir(path_sample_images)
targets_list_sample = os.listdir(path_sample_targets)

image_list_train = image_list_sample[:100]
targets_list_train = targets_list_sample[:100]

image_list_eval = path_sample_images[101:130]
targets_list_eval = path_sample_targets[101:130]

file_ids_train = [i.split('.')[0] for i in image_list_train]
file_ids_eval = [i.split('.')[0] for i in image_list_eval]

# %%
class CustomData(Dataset):
    def __init__(self, image_path, annos_path, filenames):
        self.image_path = image_path
        self.annos_path = annos_path
        self.filenames = filenames
    
    def __getitem__(self, index):

        # Get the image path and open the indexed image
        image_id = self.filenames[index] + '.jpg'
        path_image = os.path.join(self.image_path, image_id)
        img = Image.open(path_image)
        image_orig = np.array(img)

        # Move the numpy axes to the right order as accepted by Pytorch
        image_ordered = np.transpose(image_orig, (2, 0, 1))
        image_norm = image_ordered/255.0
        image = torch.Tensor(image_norm)
        # print("Image Type from Utils: ", type(image), "\n", image)

        # Load the annotations as a json
        annos_id = self.filenames[index] + '.json'
        path_anno = os.path.join(self.annos_path, annos_id)
        with open(path_anno,'r') as f:
            data = json.load(f)

        # Number of targets/labels/bounding boxes and initialize a final targets dictionary for PyTorch input
        num_targets = len([k for k in list(data.keys()) if 'item' in k])
        # print(num_targets)

        labels = [None] * num_targets
        boxes = [None] * num_targets
        # Iterate through each annotation to find the # of labels and associated bounding boxes in the image
        for j in range(1,num_targets+1):
            
            # Obtain the image label
            label = data[f'item{j}']['category_id']

            # Get the bounding box and normalize it by the width and height calculated above
            x0, y0, x1, y1 = data[f'item{j}']['bounding_box']

            # Put the bounding box information by label into a specific list
            box = [x0, y0, x1, y1]

            # Convert the list of labels into a tensor
            labels[j-1] = label

            # Convert the list of lists of bounding boxes to a torch float tensor
            boxes[j-1] = box


        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.Tensor(boxes)
    
        targets = {'boxes': boxes, 'labels': labels}

        return image, targets
    
    def __len__(self):
        return len(self.filenames)

    
def collate_fn(batch):
    return list(zip(*batch))
