import os
import tensorflow as tf
import numpy as np

"""
Parameters for each run.
    :param IMAGES_PER_CAT: Number of images to use per category; defaults to all images if not specified.
    :param BATCH_SIZE: Number of images to process in each gradient descent step.
    :param NUM_EPOCHS: (Average) number of times to rpocess each image.
    :param SEED: Random seed for run. (@brishima suspects that not all the randomness is controlled here.
    :param KEEP_PROB: Proportion of nodes in fully-connected layers to drop out.
    :param EVAL_FREQUENCY: Frequency with which we evaluate diagnostics (loss, accuracies ...). This information is
        also saved to be accessible to Tensorboard.
    :param USE_GPU: Whether to use GPU acceleration if available. Only set to False if you are receiving
        out-of-memory errors with the GPU.
    :param FLAG_RESIZE_AND_CROP: Whether to resize to a square image of IMAGE_RESIZED_SIZE and then randomly crop the
        images.
    :param FLAG_RANDOM_FLIP:
"""
IMAGES_PER_CAT = None
BATCH_SIZE = 50
NUM_EPOCHS = 100
SEED = 1234
KEEP_PROB = 0.5
EVAL_FREQUENCY = 10 # 10 will keep the epochs the same
EVAL_IMAGE_FREQUENCY = 1000
MIN_EVAL_STEP = EVAL_FREQUENCY # to skip unstable part at beginning
USE_GPU = True
CONV_REG = 1e-5
FF_REG = 1e-3

# TODO: Maybe don't use these constants directly but pass in as parameters?
FLAG_RESIZE_AND_CROP = True
IMAGE_RESIZED_SIZE = 128
IMAGE_CROPPED_SIZE = 112
FLAG_RANDOM_FLIP_LR = True
FLAG_NORMALIZE = True
FLAG_DEMEAN = True
FLAG_ADD_NOISE = False
FLAG_BATCH_NORMALIZATION = True
FLAG_TRAIN = True # not really a constant but w/e
IMAGE_MEAN = np.array([0.45834960097,0.44674252445,0.41352266842])*255

"""
Global constants. There should be no reason to modify these.
"""
TYPE = tf.float32
LABEL_TYPE = tf.int32
NUM_CHANNELS = 3
IMAGE_IMPORT_SIZE = 128
NUM_LABELS = 100

"""
Constants for logging and storing data. You are unlikely to want to modify these.
"""
PWD = os.path.dirname(__file__) + '/'
DATA_DIR = PWD + 'data/images/'
CHECKPOINT_DIRECTORY = PWD + 'checkpoints/'
LOGS_DIR = PWD + 'logs/'
TB_LOGS_DIR = PWD + 'tb_logs/'
PREDICTIONS_DIR = PWD + 'predictions/'
SUBMISSIONS_DIR = PWD + 'submissions/'

def get_categories():
    # Get list of categories, and mapping from category to index
    with open('development_kit/data/categories.txt') as f:
        categories = []
        category_to_index = {}
        for line in f:
            name, cat = line.split()
            cat = int(cat)
            categories.append(name[3:]) # skip '/./'
            category_to_index[name] = cat
    return (categories, category_to_index)
ALL_CATEGORIES, CATEGORIES_TO_INDEX = get_categories()

IMAGE_FINAL_SIZE = IMAGE_CROPPED_SIZE if FLAG_RESIZE_AND_CROP else IMAGE_IMPORT_SIZE
