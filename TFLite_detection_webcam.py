\
######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import math

PARTS = {
    0: 'NOSE',
    1: 'LEFT_EYE',
    2: 'RIGHT_EYE',
    3: 'LEFT_EAR',
    4: 'RIGHT_EAR',
    5: 'LEFT_SHOULDER',
    6: 'RIGHT_SHOULDER',
    7: 'LEFT_ELBOW',
    8: 'RIGHT_ELBOW',
    9: 'LEFT_WRIST',
    10: 'RIGHT_WRIST',
    11: 'LEFT_HIP',
    12: 'RIGHT_HIP',
    13: 'LEFT_KNEE',
    14: 'RIGHT_KNEE',
    15: 'LEFT_ANKLE',
    16: 'RIGHT_ANKLE'
}

# Mathematical function for processing heatmaps
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Class reference ziggyjosh16 from GitHub repo app-fall-dector
# https://github.com/ziggyjosh16/app-fall-detector/blob/master/person.py
class Person():
    def __init__(self, heatmap, offsets):
        self.keypoints = self.get_keypoints(heatmap, offsets)
        self.pose = self.infer_pose(self.keypoints)

    def get_keypoints(self, heatmaps, offsets, output_stride=32):
        # sigmoid activation to get scores
        scores = sigmoid(heatmaps)
        # shape from numpy returns the length of the third dimension
        num_keypoints = scores.shape[2]
        heatmap_positions = []
        offset_vectors = []
        confidences = []
        # loop through all keypoints
        for ki in range(0, num_keypoints):
            #  get 2D index from a linear index
            x, y = np.unravel_index(
                # get index for the largest value for the currnet tensor within the activated axis
                np.argmax(scores[:, :, ki]), scores[:, :, ki].shape)
            confidences.append(scores[x, y, ki])
            offset_vector = (offsets[y, x, ki],
                             offsets[y, x, num_keypoints + ki])
            heatmap_positions.append((x, y))
            offset_vectors.append(offset_vector)
        # add up two arrays
        image_positions = np.add(
            np.array(heatmap_positions) *
            output_stride,
            offset_vectors)
        # instantiate Keypoint sub object
        keypoints = [KeyPoint(i, pos, confidences[i])
                     for i, pos in enumerate(image_positions)]
        return keypoints

    def infer_pose(self, coords):
        return "Unknown"

    def get_coords(self):
        return [kp.point() for kp in self.keypoints]  # if kp.confidence > 0.8

    def get_limbs(self):
        pairs = [
            (5, 6),
            (5, 7),
            (7, 9),
            (5, 11),
            (11, 13),
            (13, 15),
            (6, 8),
            (8, 10),
            (6, 12),
            (12, 14),
            (14, 16),
            (11, 12)
        ]
        # if (self.keypoints[i].confidence > 0.8 and
        # self.keypoints[j].confidence > 0.8)
        limbs = [(self.keypoints[i].point(), self.keypoints[j].point())
                 for i, j in pairs]
        return list(filter(lambda x: x is not None, limbs))

    def get_side(self):
        lr = "middle"
        if(self.keypoints[0].get_confidence() > 0.7):
            if(self.keypoints[0].point()[0] < 257/2):
                lr = "right"
            else:
                lr = "left"
            return lr
        return
            # print(self.keypoints[0].point()[0],self.keypoints[0].point()[1])

    def confidence(self):
        return np.mean([k.confidence for k in self.keypoints])

    def to_string(self):
        return "\n".join([a.to_string() for a in self.keypoints])


class KeyPoint():
    def __init__(self, index, pos, v):
        x, y = pos
        self.x = x
        self.y = y
        self.index = index
        self.body_part = PARTS.get(index)
        self.confidence = v

    def point(self):
        return int(self.y), int(self.x)
    def get_confidence(self):
        return self.confidence
    def to_string(self):
        return 'part: {} location: {} confidence: {}'.format(
            self.body_part, (self.x, self.y), self.confidence)

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
# PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
# with open(PATH_TO_LABELS, 'r') as f:
    # labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
# if labels[0] == '???':
    # del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
# print("width:",width,"height:",height)
# height = width = 257
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    
    # Retrieve detection results
    heatmap = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    offset = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    # forward_bias = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    # backward_bias = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    print("-------")
    person = Person(heatmap, offset)
    # print(person.to_string())
    print(person.get_side())
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    # for i in range(len(scores)):
        # if ((boxes[i] > min_conf_threshold) and (boxes[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            # ymin = int(max(1,(boxes[i][0] * imH)))
            # xmin = int(max(1,(boxes[i][1] * imW)))
            # ymax = int(min(imH,(boxes[i][2] * imH)))
            # xmax = int(min(imW,(boxes[i][3] * imW)))
            #cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            # print(ymin, xmin, ymax, xmax)
            # Draw label
            # object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            # label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            # labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            # label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            # cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            # cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    # Draw framerate in corner of frame
    # cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    # cv2.imshow('Person detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
