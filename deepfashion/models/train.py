# %%
# Import base packages
import numpy as np
import json
import os
import datetime
from PIL import Image, ImageDraw

# Import AWS training package
import sagemaker
from sagemaker.pytorch import PyTorch
import boto3

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
# Set up PyTorch estimator
estimator = PyTorch(entry_point='model.py',
                    source_dir = os.getcwd(),
                    role=role,
                    framework_version='1.2.0',
                    train_instance_count=1,
                    train_instance_type='ml.p2.xlarge')

# %%
curr_time = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
estimator.fit({'train' : train_dir, 'test': val_dir}, job_name = f'{curr_time}-dfmodel')
# %%
