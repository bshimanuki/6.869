from collections import Counter

import numpy as np
import tensorflow as tf

from constants import DATA_DIR, NUM_CHANNELS, TYPE, IMAGE_RESIZE_SIZE, IMAGE_FINE_SIZE, IMAGE_MEAN, SEED


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


# Potentially leanr from www.github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/models/image/cifar10/cifar10_input.py#L222

def get_input(partition, all_categories, target_categories=[], n=None):
    """

    :param partition: String matching folder containing images. Valid values are 'test', 'train' and 'val'
    :param all_categories: All categories that images can be classified in.
    :param target_categories: Target categories. Defaults to all categories if not specified.
    :param n: Target number of examples. Defaults to infinity if not specified.
    :return:
    """
    files, labels = get_files_and_labels(partition, all_categories, target_categories, n)
    queue = tf.train.slice_input_producer([files, labels], shuffle=True)
    image = tf.read_file(queue[0])
    label = queue[1]
    image = tf.image.decode_jpeg(image, channels=NUM_CHANNELS)
    image = tf.cast(image, TYPE)
    image = tf.image.resize_images(image, [IMAGE_RESIZE_SIZE, IMAGE_RESIZE_SIZE])
    image = tf.image.random_ops.random_crop(image, [IMAGE_FINE_SIZE, IMAGE_FINE_SIZE, NUM_CHANNELS], seed=SEED)
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = image - IMAGE_MEAN # TODO: someone sanity check me here please?
    # image.set_shape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    return image, label

def get_files_and_labels(partition, all_categories, target_categories=[], n=None):
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
            if not target_categories or all_categories[cat] in target_categories:
                if n is None or num <= n:
                    files.append(DATA_DIR + name)
                    labels.append(cat)
    return files, labels

def get_size(partition, all_categories, target_categories=[], n=None):
    # TODO: better naming ...
    files, labels = get_files_and_labels(partition, all_categories, target_categories, n)
    return len(files)


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
