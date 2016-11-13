#!/usr/bin/env python3

from collections import defaultdict, Counter
from functools import reduce
import operator
import time
import sys

import tensorflow as tf
import numpy as np

IMAGES_PER_CAT = 200
BATCH_SIZE = 100
IMAGE_SIZE = 128
NUM_CHANNELS = 3
NUM_LABELS = 100
NUM_EPOCHS = 10
SEED = 1234
TYPE = tf.float32
LABEL_TYPE = tf.int32
DATA_PREFIX = 'data/images/'
EVAL_FREQUENCY = 1

with open('development_kit/data/categories.txt') as f:
    categories = []
    category_to_index = {}
    for line in f:
        name, cat = line.split()
        cat = int(cat)
        categories.append(name[3:]) # skip '/./'
        category_to_index[name] = cat

def get_files(partition, cats=[], n=None):
    cats = set(cats)
    files = []
    labels = []
    with open('development_kit/data/%s.txt' % partition) as f:
        for line in f:
            name, cat = line.split()
            num = int(''.join(filter(str.isdigit, name)))
            cat = int(cat)
            if not cats or categories[cat] in cats:
                if n is None or num <= n:
                    files.append(DATA_PREFIX + name)
                    labels.append(cat)
    train_queue = tf.train.slice_input_producer([files, labels], shuffle=True)
    images = tf.read_file(train_queue[0])
    labels = train_queue[1]
    images = tf.image.decode_jpeg(images, channels=NUM_CHANNELS)
    images = tf.cast(images, TYPE)
    images.set_shape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    return images, labels, files

def accuracy(predictions, labels, k=1):
    correct = tf.nn.in_top_k(predictions, labels, k)
    print(Counter(zip(np.argmax(predictions, 1).tolist(), labels)))
    return tf.reduce_mean(tf.cast(correct, tf.float32)).eval()

if __name__ == '__main__':
    x = tf.placeholder(TYPE, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name='input')
    y = tf.placeholder(tf.int32, shape=(BATCH_SIZE,), name='labels')

    CONV_DEPTH_1 = 64
    CONV_DEPTH_2 = 32
    FF_DEPTH_1 = 512
    POOL_STRIDE = 2
    REDUCED_IMAGE_SIZE = IMAGE_SIZE // POOL_STRIDE**2 * IMAGE_SIZE // POOL_STRIDE**2
    conv1_weights = tf.Variable(
            tf.truncated_normal(
                [5, 5, NUM_CHANNELS, CONV_DEPTH_1],
                stddev=0.1,
                seed=SEED,
                dtype=TYPE),
            name='conv1_w')
    conv1_biases = tf.Variable(tf.zeros(shape=[CONV_DEPTH_1], dtype=TYPE), name='conv1_b')
    conv2_weights = tf.Variable(
            tf.truncated_normal(
                [5, 5, CONV_DEPTH_1, CONV_DEPTH_2],
                stddev=0.1,
                seed=SEED,
                dtype=TYPE),
            name='conv2_w')
    conv2_biases = tf.Variable(tf.zeros(shape=[CONV_DEPTH_2], dtype=TYPE), name='conv2_b')
    ff1_weights = tf.Variable(
            tf.truncated_normal(
                [REDUCED_IMAGE_SIZE * CONV_DEPTH_2, FF_DEPTH_1],
                stddev=0.1,
                seed=SEED,
                dtype=TYPE),
            name='ff1_w')
    ff1_biases = tf.Variable(tf.zeros(shape=[FF_DEPTH_1], dtype=TYPE), name='ff1_b')
    ff2_weights = tf.Variable(
            tf.truncated_normal(
                [FF_DEPTH_1, NUM_LABELS],
                stddev=0.1,
                seed=SEED,
                dtype=TYPE),
            name='ff2_w')
    ff2_biases = tf.Variable(tf.zeros(shape=[NUM_LABELS], dtype=TYPE), name='ff2_b')

    def model(data):
        conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        sig = tf.nn.sigmoid(tf.nn.bias_add(conv, conv1_biases))
        pool = tf.nn.max_pool(sig, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        sig = tf.nn.sigmoid(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(sig, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [pool_shape[0], reduce(operator.mul, pool_shape[1:], 1)])
        hidden = tf.nn.sigmoid(tf.matmul(reshape, ff1_weights) + ff1_biases)
        return tf.matmul(hidden, ff2_weights) + ff2_biases

    cats = ['abbey', 'playground']
    train_data, train_labels, train_files = get_files('train', cats, n=IMAGES_PER_CAT)
    train_size = len(train_files)
    logits = model(x)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    regularizers = tf.nn.l2_loss(ff1_weights) + tf.nn.l2_loss(ff1_biases) + \
            tf.nn.l2_loss(ff2_weights) + tf.nn.l2_loss(ff2_biases)
    loss += 1e-6 * regularizers

    batch = tf.Variable(0, dtype=TYPE)
    learning_rate = tf.train.exponential_decay(
            0.01,                # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            train_size,          # Decay step.
            0.9,                # Decay rate.
            staircase=True)
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.25).minimize(loss, global_step=batch)

    train_prediction = tf.nn.softmax(logits)

    batch_data, batch_labels = tf.train.batch(
            [train_data, train_labels],
            batch_size=BATCH_SIZE)

    start_time = time.time()
    config = tf.ConfigProto()
    # config.operation_timeout_in_ms = 2000
    with tf.Session(config=config) as sess:
        # Run all the initializers to prepare the trainable parameters.
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print('Initialized!')

        # Loop through training steps.
        for step in range(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
            _data, _labels = sess.run([batch_data, batch_labels])

            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {x: _data,
                    y: _labels}
            # Run the optimizer to update weights.
            sess.run(optimizer, feed_dict=feed_dict)

            # print some extra information once reach the evaluation frequency
            if step % EVAL_FREQUENCY == 0:
                # fetch some extra nodes' data
                l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                        feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                        (step, float(step) * BATCH_SIZE / train_size,
                            1000 * elapsed_time / EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch accuracy: %.1f%%' % (100 * accuracy(predictions, _labels)))
                sys.stdout.flush()

        coord.request_stop()
        coord.join(threads)
