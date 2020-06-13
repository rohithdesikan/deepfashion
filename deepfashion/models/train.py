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

# %%
# Set the file paths
curr_dir = os.getcwd()
path_sample = os.path.abspath(os.path.join(curr_dir, os.pardir, 'data', 'sample_data'))
print(path_sample)

# Set up sagemaker session
sagemaker_session = sagemaker.Session(default_bucket = 'rohithdesikan-deepfashion')

# Get default bucket
bucket = sagemaker_session.default_bucket()
print(bucket)

# Get role
role = sagemaker.get_execution_role()
print(role)

# %%
# set prefix, a descriptive name for the S3 directory
prefix = 'deepfashion_sample'

# upload all data to S3
# sagemaker_session.upload_data(path_sample, bucket=bucket, key_prefix=prefix)

# %%
bucket_name = 's3://rohithdesikan-deepfashion/deepfashion_sample'
print(bucket_name)

Image.open('s3://rohithdesikan-deepfashion/deepfashion_sample/image/000001.jpg')

# %%

estimator = PyTorch(entry_point='model.py',
                    role=role,
                    framework_version='1.2.0',
                    train_instance_count=2,
                    train_instance_type='ml.p2.xlarge',
                    output_path='s3://rohithdesikan-deepfashion/deepfashion_sample/output',
                    hyperparameters={
                        'epochs': 5,
                    })

# %%
# TO DO: FIGURE OUT HOW TO GET THE LIST OF FILES WITHIN AN S3 BUCKET SO THAT THEY CAN BE USED TO FIGURE OUT THE FILENAMES THAT NEED TO BE OPENED. LOOK AT THE UDACITY COURSE TO SEE HOW BEST TO LOOK AT FILE PATHS ON AWS
estimator.fit({'bucket_name': bucket_name})

# %%
