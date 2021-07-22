# %%
# Import base packages
import numpy as np
import argparse
import json
import logging
import os
import sys
from PIL import Image

# Import AWS training package
import sagemaker
from sagemaker.pytorch import PyTorch
import boto3

# %%
curr_dir = os.getcwd()
path_raw = os.path.abspath(os.path.join(curr_dir, os.pardir, os.pardir, 'data', 'processed'))

local_path_train = os.path.join(path_raw, 'train')
local_path_val = os.path.join(path_raw, 'validation')
local_path_train_images = os.path.join(local_path_train, 'image')
local_path_train_annos = os.path.join(local_path_train, 'annos')
local_path_val_images = os.path.join(local_path_val, 'image')
local_path_val_annos = os.path.join(local_path_val, 'annos')

train_image_list = os.listdir(local_path_train_images)
train_annos_list = os.listdir(local_path_train_annos)
val_image_list = os.listdir(local_path_val_images)
val_annos_list = os.listdir(local_path_val_annos)

file_ids_train = [i.split('.')[0] for i in train_image_list]
file_ids_val = [i.split('.')[0] for i in val_image_list]

# %%
# Set the file paths
curr_dir = os.getcwd()
path_sample = os.path.abspath(os.path.join(curr_dir, os.pardir, 'data', 'sample_data'))
print(path_sample)

# Set up sagemaker session
sagemaker_session = sagemaker.Session(default_bucket = 'rohithdesikan-deepfashion')

# Get default bucket
bucket_name = sagemaker_session.default_bucket()
print(bucket_name)

# Get role
role = sagemaker.get_execution_role()
print(role)

# set prefix, a descriptive name for the S3 directory
prefix = 'deepfashion_sample'

# upload all data to S3
# sagemaker_session.upload_data(path_sample, bucket=bucket, key_prefix=prefix)

# Set up file paths on S3
bucket_path = 's3://rohithdesikan-deepfashion/deepfashion-sample'
train_dir = 's3://rohithdesikan-deepfashion/deepfashion-sample/train'
val_dir = 's3://rohithdesikan-deepfashion/deepfashion-sample/val'
train_images_path = 's3://rohithdesikan-deepfashion/deepfashion-sample/train/image'
train_targets_path = 's3://rohithdesikan-deepfashion/deepfashion-sample/train/annos'
val_images_path = 's3://rohithdesikan-deepfashion/deepfashion-sample/val/image'
val_targets_path = 's3://rohithdesikan-deepfashion/deepfashion-sample/val/annos'
output_location = 's3://rohithdesikan-deepfashion/outputs'

# %%
manifest_json = []
for file_id in file_ids_train:
    manifest_dict = {'source-ref' = train_images_path + rf"/{file_id}"}




# %%
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

