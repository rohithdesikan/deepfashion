# %%
# Import base packages
import numpy as np
import json
import os
import datetime
from PIL import Image, ImageDraw

# Import AWS training package
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

# %%
# Set the file paths
curr_dir = os.getcwd()
path_sample = os.path.join(curr_dir, 'deepfashion', 'data', 'sample_data')

# Set up sagemaker session
sagemaker_session = sagemaker.Session(default_bucket = 'rohithdesikan-deepfashion')

# Get default bucket
# bucket = sagemaker_session.default_bucket()
# print(bucket)

# Get role
role = sagemaker.get_execution_role()
print(role)

# set prefix, a descriptive name for a directory
# prefix = 'deepfashion_sample'

# upload all data to S3
# bucket_data = sagemaker_session.upload_data(path_sample, bucket=bucket, key_prefix=prefix)
bucket_data = 's3://rohithdesikan-deepfashion/deepfashion_sample'
print(bucket_data)
# %%

estimator = PyTorch(entry_point='deepfashion.py',
                    role=role,
                    framework_version='1.2.0',
                    train_instance_count=2,
                    train_instance_type='ml.m4.xlarge',
                    hyperparameters={
                        'epochs': 10,
                    })

# %%
estimator.fit({'training': bucket_data})

# %%
