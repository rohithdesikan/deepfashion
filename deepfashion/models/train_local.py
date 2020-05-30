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
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Import AWS training package
import sagemaker

# Import local packages
from model import TransformData, collate_fn, FasterRCNN

print("Torch Version: ", torch.__version__, "TorchVision Version: ", torchvision.__version__)


# %%
# Set the file paths
curr_dir = os.getcwd()
path_sample = os.path.abspath(os.path.join(curr_dir, os.pardir, 'data', 'sample_data'))
path_raw = os.path.abspath(os.path.join(curr_dir, os.pardir, os.pardir, 'data', 'raw'))

path_train = os.path.join(path_raw, 'train')
path_val = os.path.join(path_raw, 'validation')
path_train_images = os.path.join(path_train, 'image')
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

# target_height, target_width = 500, 400

# %%
# Load the dataset and the data loader
dataset = TransformData(path_sample_images, path_sample_targets, file_ids_train)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True, num_workers = 6, collate_fn = collate_fn)

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
    for batch_images, batch_targets in data_loader:
        i += 1
        images = list(batch_images.to(device))
        targets = list(batch_targets.to(device))


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