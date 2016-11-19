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
NUM_EPOCHS = 30
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

def weight_variable(shape, name=None):
    return tf.Variable(
            tf.truncated_normal(
                shape,
                stddev=0.1,
                seed=SEED,
                dtype=TYPE),
            name=name)

def bias_variable(shape, name=None):
    return tf.Variable(tf.zeros(shape=shape, dtype=TYPE), name=name)

def conv_layer(input_layer, depth, window, name=None, variables=None):
    assert(input_layer.get_shape().ndims == 4)
    w_name = None if name is None else name + '_w'
    b_name = None if name is None else name + '_b'
    w = weight_variable([window, window, input_layer.get_shape().as_list()[-1], depth], w_name)
    b = bias_variable([depth], b_name)
    conv = tf.nn.conv2d(input_layer, w, strides=[1, 1, 1, 1], padding='SAME') + b
    act = tf.nn.sigmoid(conv)
    pool = tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    if variables is not None:
        variables['conv_w'].append(w)
        variables['conv_b'].append(b)
    return pool

def ff_layer(input_layer, depth, name=None, activation=True, variables=None):
    assert(input_layer.get_shape().ndims == 2)
    w_name = None if name is None else name + '_w'
    b_name = None if name is None else name + '_b'
    w = weight_variable([input_layer.get_shape().as_list()[-1], depth], w_name)
    b = bias_variable([depth], b_name)
    hidden = tf.matmul(input_layer, w) + b
    if activation:
        hidden = tf.nn.sigmoid(hidden)
    if variables is not None:
        variables['ff_w'].append(w)
        variables['ff_b'].append(b)
    return hidden

def conv_to_ff_layer(conv):
    shape = conv.get_shape().as_list()
    reshape = tf.reshape(conv, [shape[0], reduce(operator.mul, shape[1:], 1)])
    return reshape

def model(data):
    variables = defaultdict(list)
    conv = conv_layer(data, depth=64, window=5, name='conv1', variables=variables)
    conv = conv_layer(conv, depth=32, window=5, name='conv2', variables=variables)
    reshape = conv_to_ff_layer(conv)
    hidden = ff_layer(reshape, depth=512, name='ff1', variables=variables)
    output = ff_layer(hidden, depth=NUM_LABELS, name='ff2', activation=False, variables=variables)
    return output, variables

if __name__ == '__main__':
    x = tf.placeholder(TYPE, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name='input')
    y = tf.placeholder(tf.int32, shape=(BATCH_SIZE,), name='labels')

    cats = ['abbey', 'playground']
    train_data, train_labels, train_files = get_files('train', cats, n=IMAGES_PER_CAT)
    train_size = len(train_files)
    logits, variables = model(x)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    regularizers = sum(map(tf.nn.l2_loss, variables['ff_w'] + variables['ff_b']))
    loss += 1e-6 * regularizers

    batch = tf.Variable(0, dtype=TYPE)
    learning_rate = tf.train.exponential_decay(
            0.01,                # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            train_size,          # Decay step.
            0.9,                 # Decay rate.
            staircase=False)
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
                l, r, lr, predictions = sess.run([loss, regularizers, learning_rate, train_prediction],
                        feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                        (step, float(step) * BATCH_SIZE / train_size,
                            1000 * elapsed_time / EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, regularizers: %.6g learning rate: %.6f' % (l, r, lr))
                print('Minibatch accuracy: %.1f%%' % (100 * accuracy(predictions, _labels)))
                sys.stdout.flush()

        coord.request_stop()
        coord.join(threads)
