import keras
import pickle
from videotest import VideoTest
import pandas as pd
import sys
sys.path.append("..")
from ssd import SSD300 as SSD

input_shape = (300, 300, 3)

# Change this if you run with other classes than VOC
class_names = ['SS','RN','RM','SM','SY','TO','YY', 'HU']
NUM_CLASSES = len(class_names)

model = SSD(input_shape, num_classes=NUM_CLASSES)

# Change this path if you want to use your own trained weights
# model.load_weights('../weights_SSD300.hdf5')
# model.load_weights('../checkpoints/weights.04-2.58.hdf5')
model.load_weights('../checkpoints//weights.10-5.21.hdf5', by_name=True)

vid_test = VideoTest(class_names, model, input_shape)
vid_test.run_img(img_path='../dataset/JPEGImages/', conf_thresh=0.75)

