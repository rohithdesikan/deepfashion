# %%
# Import base packages
import os

# Import AWS packages
import boto3
import sagemaker

# %%
# Set the file paths
curr_dir = os.getcwd()
path_sample = os.path.join(curr_dir, 'deepfashion', 'data', 'sample_data')

# %%
# Set up sagemaker session
sagemaker_session = sagemaker.Session(default_bucket = 'rohithdesikan-deepfashion')

# Get default bucket
bucket = sagemaker_session.default_bucket()
print(bucket)

# Get role
role = sagemaker.get_execution_role()
print(role)

# %%
# set prefix, a descriptive name for a directory  
prefix = 'deepfashion_sample'

# upload all data to S3
bucket_data = sagemaker_session.upload_data(path_sample, bucket=bucket, key_prefix=prefix)

# %%
