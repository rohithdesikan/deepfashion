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
from model import TransformData, collate_fn

print("Torch Version: ", torch.__version__, "TorchVision Version: ", torchvision.__version__)


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

# target_height, target_width = 500, 400

# %%
# Load the dataset and the data loader
dataset = TransformData(path_sample_images, path_sample_annos, file_ids)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True, num_workers = 6, collate_fn = collate_fn)

# batch_images, batch_annos = next(iter(data_loader))
# batch_list_images = list(batch_images)
# batch_list_annos = list(batch_annos)

# print(len(batch_list_images), type(batch_list_images), batch_list_images[0].shape, batch_list_images[0])
# print(len(batch_list_annos), type(batch_list_annos), batch_list_annos[0])

# %%
# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# %%
# Load the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 14)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr = 0.001)
model.to(device)

# %%
# Training
num_epochs = 2
for epoch in range(1, num_epochs):
    print("Epoch: ", epoch)

    model.train()

    i = 0
    for batch_images, batch_annos in data_loader:
        i += 1
        images = list(batch_images.to(device))
        annos = list(batch_annos.to(device))


        loss_dict = model(images, annos)
        losses = sum(loss for loss in loss_dict.values())

        model.zero_grad()
        losses.backward()
        optimizer.step()

        print("Iteration #:  ", i/20, "Loss: ", losses)

# %%
# Save model:
torch.save(model.state_dict(), f"experiments/model_{datetime.datetime.now().strftime('%D')}.pt")

# DOWNLOAD THE OLD MODEL.PT AND LOOK AT THE OUTPUT