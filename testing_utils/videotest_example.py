import keras
import pickle
from videotest import VideoTest

import sys
sys.path.append("..")
from ssd import SSD300 as SSD

input_shape = (300, 300, 3)

# Change this if you run with other classes than VOC
class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "meteor"];
NUM_CLASSES = len(class_names)

model = SSD(input_shape, num_classes=NUM_CLASSES)

# Change this path if you want to use your own trained weights
# model.load_weights('../weights_SSD300.hdf5')
# model.load_weights('../checkpoints/weights.04-2.58.hdf5')
model.load_weights('../checkpoints/weights.09-2.90.hdf5')

vid_test = VideoTest(class_names, model, input_shape)

# To test on webcam 0, remove the parameter (or change it to another number
# to test on that webcam)
#vid_test.run('path/to/your/video.mkv')
# vid_test.run('../videos/708213662.mp4')
# vid_test.run('../videos/722729280.mp4')
# vid_test.run('../videos/792677823.mp4')
# vid_test.run('../videos/348456619.mp4')
vid_test.run('D://Develop/stair/train/000005.mpg', conf_thresh=0.85)
