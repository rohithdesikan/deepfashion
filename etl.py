# %%
# Import base packages
import numpy as np
# import io
import os
import json
from PIL import Image, ImageDraw
import skimage

import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util, dataset_util
from object_detection.utils import visualization_utils as vis_util

flags = tf.compat.v1.app.flags
# flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
# flags.DEFINE_string('image_dir', '', 'Path to the image directory')
# flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

# %%
# Resize images and output to a processed folder
def resize_images(path_input_images,
                path_input_annos,
                path_output_images,
                path_output_annos,
                image_fns,
                annos_fns,
                target_height, 
                target_width):


    # Iterate through the images
    for i, (image_fn, anno_fn) in enumerate(zip(image_fns, annos_fns)):
    
        # Output image size to a numpy array
        path_image = os.path.join(path_input_images, image_fn)
        img = Image.open(path_image)
        image_orig = np.array(img)

        # Get the original size of the array
        h_orig, w_orig = image_orig.shape[:2]

        # Transform the array to a new target size specified above
        image_scaled = skimage.transform.resize(image_orig, (target_height, target_width))
        image_uint8 = skimage.util.img_as_ubyte(image_scaled)

        # Output the new image to a processed folder
        out_image_path = os.path.join(path_output_images, image_fn)
        skimage.io.imsave(out_image_path, image_uint8)


        # Load the annotations as a json
        path_anno = os.path.join(path_input_annos, anno_fn)
        with open(path_anno,'r') as infile:
            data = json.load(infile)

        # Number of targets/labels/bounding boxes and initialize a final targets dictionary for input
        num_targets = len([k for k in list(data.keys()) if 'item' in k])

        # Iterate through each annotation to find the # of labels and associated bounding boxes in the image
        for j in range(1,num_targets+1):
            
            # Obtain the image label
            label = data[f'item{j}']['category_id']

            # Get the bounding box and normalize it by the width and height calculated above
            x0, y0, x1, y1 = data[f'item{j}']['bounding_box']

            # Calculate new bounding boxes with the target size
            x0_new = x0 * target_width / w_orig
            y0_new = y0 * target_height / h_orig
            x1_new = x1 * target_width / w_orig
            y1_new = y1 * target_height / h_orig
            
            # Reassign the new bounding boxes
            data[f'item{j}']['bounding_box'] = [x0_new, y0_new, x1_new, y1_new]
        
        # Set the annotations path
        out_annos_path = os.path.join(path_output_annos, anno_fn)
        
        # Dump a new json file
        with open(out_annos_path, 'w') as outfile:
            json.dump(data, outfile)

# %%
# Generate a TFRecord file
def generate_tfrecord(path_input_images_processed, path_input_annos_processed, image_fn, annos_fn):

    # Generate an image path and read it in
    path_image = os.path.join(path_input_images_processed, image_fn)
    encoded_path_image = bytes(path_image.encode('utf-8'))
    image = Image.open(path_image)

    # Get height and width for TF Record Info
    width, height = image.size

    with tf.io.gfile.GFile(path_image, 'rb') as im_fn:
        jpg_file = im_fn.read()
    
    # Encode the image as a jpg, set the image format and find the image id
    encoded_jpg = bytes(jpg_file)
    image_format = b'jpg'
    image_id = image_fn.split('.')[0]
    encoded_image_id = bytes(image_id.encode('utf-8'))


    # Load the annotations as a json
    path_anno = os.path.join(path_input_annos_processed, annos_fn)
    with open(path_anno,'r') as infile:
        data = json.load(infile)
    
    # Number of targets/labels/bounding boxes and initialize a final targets dictionary for input
    num_targets = len([k for k in list(data.keys()) if 'item' in k])
    
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    class_labels = []
    class_names = []
    # Iterate through each annotation to find the # of labels and associated bounding boxes in the image
    for i in range(1,num_targets+1):
        
        # Obtain the image label
        label = data[f'item{i}']['category_id']
        name = data[f'item{i}']['category_name']


        # Get the bounding box and normalize it by the width and height calculated above
        x0, y0, x1, y1 = data[f'item{i}']['bounding_box']

        # Calculate new bounding boxes with the target size
        xmins.append(x0/width)
        ymins.append(y0/height)
        xmaxs.append(x1/width)
        ymaxs.append(y1/height)

        class_labels.append(label)
        class_names.append(name.encode('utf-8'))
    

    tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(encoded_path_image),
    'image/source_id': dataset_util.bytes_feature(encoded_image_id),
    'image/encoded': dataset_util.bytes_feature(encoded_jpg),
    'image/format': dataset_util.bytes_feature(image_format),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(class_names),
    'image/object/class/label': dataset_util.int64_list_feature(class_labels)
    }))

    return tf_example
