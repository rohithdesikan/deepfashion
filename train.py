#%%
# Import base packages
import numpy as np
import os 
import json
import time
from IPython.display import display

# Import visualization packages
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns

# Import ML packages
import skimage
import keras
import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util, dataset_util
from object_detection.utils import visualization_utils as vis_util

import etl

# %%
# Set the file paths
curr_dir = os.getcwd()

path_train = os.path.join(curr_dir,'datasets','train')
path_val = os.path.join(curr_dir,'datasets','validation')
path_train_images = os.path.join(path_train, 'image')
path_train_annos = os.path.join(path_train, 'annos')

path_train_processed = os.path.join(curr_dir,'datasets','processed_train')
path_val_processed = os.path.join(curr_dir,'datasets','processed_validation')
path_train_images_processed = os.path.join(path_train_processed, 'image')
path_train_annos_processed = os.path.join(path_train_processed, 'annos')

image_list = os.listdir(path_train_images)
annos_list = os.listdir(path_train_annos)
num_images = len(image_list)

image_list_small = image_list[:100]
annos_list_small = annos_list[:100]

image_list_eval = image_list[101:130]
annos_list_eval = annos_list[101:130]

target_height, target_width = 500, 400

# %%
# Set up a single image to view
#region
trial_image = '000013'

# Create image path and read in image
path_image = os.path.join(path_train_images_processed, f'{trial_image}.jpg')
image = Image.open(path_image)
width, height = image.size
arr = np.array(image)
plt.imshow(image)

# Create annotations path and read it in
path_annos = os.path.join(path_train_annos_processed, f'{trial_image}.json')
with open(path_annos,'r') as f:
    data = json.load(f)

num_items = len([k for k in list(data.keys()) if 'item' in k])

for i in range(1,num_items+1):
    label = data[f'item{i}']['category_id']
    x0,y0,x1,y1 = data[f'item{i}']['bounding_box']

    draw = ImageDraw.Draw(image)
    draw.rectangle([(x0, y0), (x1, y1)], outline='red')

plt.imshow(image)
#endregion

# %%
# Resize the images
etl.resize_images(path_train_images,
                path_train_annos,
                path_train_images_processed,
                path_train_annos_processed,
                image_list_eval,
                annos_list_eval,
                target_height,
                target_width)


# %%
# Create a tfrecord of the training set
writer = tf.compat.v1.python_io.TFRecordWriter('datasets/val.tfrecord')
for image_fn, anno_fn in zip(image_list_eval, annos_list_eval):
    tf_example = etl.generate_tfrecord(path_train_images_processed, path_train_annos_processed, image_fn, anno_fn)
    writer.write(tf_example.SerializeToString())

writer.close()

# %%
def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name, 
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model

