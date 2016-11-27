import os
import tensorflow as tf

IMAGES_PER_CAT = 200
BATCH_SIZE = 10 # TODO: return to 200
IMAGE_SIZE = 128
NUM_CHANNELS = 3
NUM_LABELS = 100
NUM_EPOCHS = 30
SEED = 1234
KEEP_PROB = 0.5
EVAL_FREQUENCY = 10
USE_GPU = True
TYPE = tf.float32
LABEL_TYPE = tf.int32

PWD = os.path.dirname(__file__) + '/'
DATA_DIR = PWD + 'data/images/'
CHECKPOINT_DIRECTORY = PWD + 'checkpoints/'
LOGS_DIR = PWD + 'logs/'
TB_LOGS_DIR = PWD + 'tb_logs/'
