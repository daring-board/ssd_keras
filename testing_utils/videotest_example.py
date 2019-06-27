import keras
import pickle
from videotest import VideoTest
import pandas as pd
import sys
sys.path.append("..")
from ssd import SSD300 as SSD

input_shape = (300, 300, 3)

# Change this if you run with other classes than VOC
class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
                "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"];
NUM_CLASSES = len(class_names)

model = SSD(input_shape, num_classes=NUM_CLASSES)

# Change this path if you want to use your own trained weights
model.load_weights('../weights_SSD300.hdf5')
# model.load_weights('../checkpoints/weights.04-2.58.hdf5')
# model.load_weights('../checkpoints/weights.09-2.90.hdf5')

vid_test = VideoTest(class_names, model, input_shape)

# To test on webcam 0, remove the parameter (or change it to another number
# to test on that webcam)
# vid_test.run(0)
#vid_test.run('../movie/play_tennis.mp4')
#vid_test.run('../movie/VideoOfPeopleWalking.mp4')
vid_test.run_img()
# df = pd.read_csv('./train_list.csv', header=None)
# with open('./ssd_result.csv', 'w') as f:
#     f.write('id, ret\n')
#     for row in df[: 200].iterrows():
#         print(row[1][0])
#         flags = vid_test.run('D://Develop/stair/train/%s'%row[1][0], conf_thresh=0.85)
#         f.write('%s, %s\n'%(row[1][0], ':'.join(map(str, flags))))
# #        break
