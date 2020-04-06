#%%
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
import os 
import matplotlib.pyplot as plt
import json
import time
import keras
from keras import tensorflow

curr_dir = os.getcwd()
print(curr_dir)
path_train = os.path.join(curr_dir,'datasets','train')
path_val = os.path.join(curr_dir,'datasets','validation')