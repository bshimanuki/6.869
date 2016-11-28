from collections import Counter

import numpy as np
import tensorflow as tf

from constants import ALL_CATEGORIES, DATA_DIR, NUM_CHANNELS, TYPE, IMAGE_SIZE

def get_files(partition, target_categories=[], n=None):
    """

    :param partition: String matching folder containing images. Valid values are 'test', 'train' and 'val'
    :param target_categories: Target categories. Defaults to all categories if not specified.
    :param n: Target number of examples. Defaults to infinity if not specified.
    :return:
    """
    target_categories = set(target_categories)
    files = []
    labels = []
    with open('development_kit/data/%s.txt' % partition) as f:
        for line in f:
            name, cat = line.split()
            num = int(''.join(filter(str.isdigit, name)))
            cat = int(cat)
            """
            Current example is used if:
                (1) no categories were specified, or if its category is in the list of target categories `cats`.
                (2) `n` is not specified, or the number of examples so far is less than the target `n`
            """
            if not target_categories or ALL_CATEGORIES[cat] in target_categories:
                if n is None or num <= n:
                    files.append(DATA_DIR + name)
                    labels.append(cat)
    queue = tf.train.slice_input_producer([files, labels], shuffle=True)
    images = tf.read_file(queue[0])
    labels = queue[1]
    images = tf.image.decode_jpeg(images, channels=NUM_CHANNELS)
    images = tf.cast(images, TYPE)
    images.set_shape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    return images, labels, files


def accuracy(predictions, labels, k=1):
    """Determines the accuracy of the predictions, and prints tally of (top_prediction, label).

    A prediction is considered accurate if the label is among the top k predictions.

    :param predictions: (batch_size, classes) tensor of predictions, with each entry corresponding to the probability
        that an example corresponds to a given class.
    :param labels: batch_size vector of class ids.
    :param k:
    :return: Proportion of accurate predictions, as a 0-dimensional tensor.
    """
    correct = tf.nn.in_top_k(predictions, labels, k)
    # print('\t', Counter(zip(np.argmax(predictions, 1).tolist(), labels)))
    return tf.reduce_mean(tf.cast(correct, tf.float32))
