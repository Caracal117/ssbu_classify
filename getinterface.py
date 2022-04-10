import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import warnings
import cv2  # still used to save images out
from decord import VideoReader
from decord import cpu, gpu
import math
import sys
import pandas as pd
import video2frame as v2f
# warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
# tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


if __name__ == '__main__':

# Initiate argument
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  matplotlib.use('TkAgg')
  # Set video path
  directory = v2f.video_to_frames(video_path='clip2_test.mp4', frames_dir='test_frames', overwrite=True, every=1)
  # directory = "D:/smash/dect/tf-detection"
  IMAGE_PATHS = [directory + "\\" + f for f in os.listdir(directory) if f[-4:] in ['.jpg','.png','.bmp']]
  PATH_TO_SAVED_MODEL = "D:/smash/dect/tf-detection/exported-models/class_frcnn_2_20k/saved_model/"
  PATH_TO_LABELS = "D:/smash/dect/tf-detection/dataset/classify_2/label_map.pbtxt"
  IMAGE_WIDTH = 1280
  IMAGE_HEIGHT = 720
  SCORE_THRESH = 0.3
  print('Loading model...', end='')
  start_time = time.time()
  # Load saved model and build the detection function
  detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
  end_time = time.time()
  elapsed_time = end_time - start_time
  print('Done! Took {} seconds'.format(elapsed_time))
  
  category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                      use_display_name=True)
  boxes = []
  classes = []
  scores = []
  i = 0
  for image_path in IMAGE_PATHS:
      # print('Running inference for {}... '.format(image_path), end='')

      image_np = load_image_into_numpy_array(image_path)
      # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
      input_tensor = tf.convert_to_tensor(image_np)
      # The model expects a batch of images, so add an axis with `tf.newaxis`.
      input_tensor = input_tensor[tf.newaxis, ...]

      # input_tensor = np.expand_dims(image_np, 0)
      detections = detect_fn(input_tensor)

      # All outputs are batches tensors.
      # Convert to numpy arrays, and take index [0] to remove the batch dimension.
      # We're only interested in the first num_detections.
      num_detections = int(detections.pop('num_detections'))
      detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
      detections['num_detections'] = num_detections
      # detection_classes should be ints.
      detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

      image_np_with_detections = image_np.copy()

      viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=2,
            min_score_thresh=.30,
            agnostic_mode=False)

      plt.figure()
      plt.imsave("{:010d}.jpg".format(i),image_np_with_detections)
      print('Done')
      i += 1
  # plt.show()
      # convert to pixel co-ordinates
  #     detection_boxes = detections['detection_boxes']
  #     detection_scores = detections['detection_scores']
  #     detection_classes = detections['detection_classes']
  #     detection_boxes[:, (0, 2)] *= IMAGE_HEIGHT
  #     detection_boxes[:, (1, 3)] *= IMAGE_WIDTH
  #     cond = (detection_scores >= SCORE_THRESH) & ((detection_scores == max(detection_scores)) )
  #     boxes.append(np.round(detection_boxes[cond,:]).astype(int))
  #     classes.append(np.round(detection_classes[cond]).astype(int))
  #     scores.append(np.round(detection_scores[cond]*100).astype(int))
  # dict = {'BOX': boxes, 'CLASSS': classes, 'SCORE': scores}
  # df = pd.DataFrame(dict)
  # df.to_csv('frcnn_test.csv')
  print("All Done") 