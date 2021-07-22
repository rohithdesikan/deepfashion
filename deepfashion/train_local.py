# %%
# Import base packages
import numpy as np
import json
import os
import datetime
from PIL import Image, ImageDraw

# Import PyTorch packages
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim

# Import AWS training package
import sagemaker

# Import local packages
from model import TransformData, collate_fn
# from model import TransformData, collate_fn, FasterRCNN

print("Torch Version: ", torch.__version__, "TorchVision Version: ", torchvision.__version__)


# %%
# Set the file paths
curr_dir = os.getcwd()
path_raw = os.path.abspath(os.path.join(curr_dir, os.pardir, os.pardir, 'data', 'processed'))

local_path_train = os.path.join(path_raw, 'train')
local_path_val = os.path.join(path_raw, 'validation')
local_path_train_images = os.path.join(local_path_train, 'image')
local_path_train_targets = os.path.join(local_path_train, 'annos')
local_path_val_images = os.path.join(local_path_val, 'image')
local_path_val_targets = os.path.join(local_path_val, 'annos')

train_image_list = os.listdir(local_path_train_images)
train_targets_list = os.listdir(local_path_train_targets)
val_image_list = os.listdir(local_path_val_images)
val_targets_list = os.listdir(local_path_val_targets)

file_ids_train = [i.split('.')[0] for i in train_image_list]
file_ids_val = [i.split('.')[0] for i in val_image_list]

# target_height, target_width = 500, 400

# %%
# Load the dataset and the data loader
dataset = TransformData(local_path_train_images, local_path_train_targets, file_ids_train)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers = 6, collate_fn = collate_fn)

batch_images, batch_targets = next(iter(data_loader))
batch_list_images = list(batch_images)
batch_list_targets = list(batch_targets)

print(len(batch_list_images), type(batch_list_images), batch_list_images[0].shape, batch_list_images[0])
print(len(batch_list_targets), type(batch_list_targets), batch_list_targets[0])

# %%
# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# %%
# Load the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 14)

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr = 0.001)

# %%
# Set up the Pytorch model in a FasterRCNN class and set all parameters to be trainable
class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.frcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = self.frcnn_model.roi_heads.box_predictor.cls_score.in_features
        self.frcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 14)

    def forward(self, images, targets):
        x = self.frcnn_model(images, targets)

        return x

# %%
model = FasterRCNN()
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr = 0.001)

# %%
# Training
num_epochs = 2
for epoch in range(1, num_epochs):
    print("Epoch: ", epoch)

    model.train()

    i = 0
    for idx, (batch_images, batch_targets) in enumerate(data_loader):
        i += 1

        images = list(img.to(device) for img in batch_images)
        print(type(images), len(images))
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]
        print(type(targets), len(targets))


        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        model.zero_grad()
        losses.backward()
        optimizer.step()

        print("Iteration #:  ", i/20, "Loss: ", losses)

# %%
# Save model:
torch.save(model.state_dict(), f"experiments/model_{datetime.datetime.now().strftime('%D')}.pt")

# DOWNLOAD THE OLD MODEL.PT AND LOOK AT THE OUTPUT