import os
import tensorflow as tf

IMAGES_PER_CAT = 200 # TODO: return to None
BATCH_SIZE = 100 # TODO: return to 200
IMAGE_SIZE = 128
NUM_CHANNELS = 3
NUM_LABELS = 100
NUM_EPOCHS = 100
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
