import os
import tensorflow as tf
import numpy as np

IMAGES_PER_CAT = None # TODO: return to None
BATCH_SIZE = 200 # TODO: return to 200
IMAGE_IMPORT_SIZE = 128
IMAGE_RESIZE_SIZE = 256
IMAGE_FINE_SIZE = 224
IMAGE_MEAN = np.array([0.45834960097,0.44674252445,0.41352266842])
NUM_CHANNELS = 3
NUM_LABELS = 100
NUM_EPOCHS = 100
SEED = 1234
KEEP_PROB = 0.5
EVAL_FREQUENCY = 1
USE_GPU = False
TYPE = tf.float32
LABEL_TYPE = tf.int32

PWD = os.path.dirname(__file__) + '/'
DATA_DIR = PWD + 'data/images/'
CHECKPOINT_DIRECTORY = PWD + 'checkpoints/'
LOGS_DIR = PWD + 'logs/'
TB_LOGS_DIR = PWD + 'tb_logs/'
