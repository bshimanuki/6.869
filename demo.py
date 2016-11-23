#!/usr/bin/env python3

from collections import defaultdict, Counter
from functools import reduce
import operator
import time
import sys
import os

import logger
import subprocess
import argparse

# all prints

timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
full_log = logger.createLogger("logs/full_output__" + timestamp + ".log", "all outputs", True)
print = full_log.info
hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode('UTF-8')[:-1]
param_log_name = "logs/save" + hash + "__" + timestamp +".log"
# param_log = logger.createLogger(param_log_name, "parameters", True)

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
CHECKPOINT_DIRECTORY = 'checkpoints'

# Get list of categories, and mapping from category to index
with open('development_kit/data/categories.txt') as f:
    categories = []
    category_to_index = {}
    for line in f:
        name, cat = line.split()
        cat = int(cat)
        categories.append(name[3:]) # skip '/./'
        category_to_index[name] = cat

def get_files(partition, cats=[], n=None):
    """

    :param partition: String matching folder containing images. Valid values are 'test', 'train' and 'val'
    :param cats: Target categories. Defaults to all categories if not specified.
    :param n: Target number of examples. Defaults to infinity if not specified.
    :return:
    """
    cats = set(cats)
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
            if not cats or categories[cat] in cats:
                if n is None or num <= n:
                    files.append(DATA_PREFIX + name)
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
    :return: Proportion of accurate predictions.
    """
    correct = tf.nn.in_top_k(predictions, labels, k)
    print('\t', Counter(zip(np.argmax(predictions, 1).tolist(), labels)))
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

def conv_layer(input_layer, depth, window, pool=None, name=None, variables=None):
    """Construct a convolutional layer which takes input_layer as input.

    input_layer -> output
    (batch_size, height, width, input_depth) -> (batch_size, height, width, depth)

    :param input_layer: input tensor
    :param depth: number of convolution images
    :param window: size of convolutional kernel (side length)
    :param pool: None for no pooling, stride length for pooling
    :param name:
    :param variables: dict with keys conv_w and conv_b to add weight and bias variables to
    """
    assert(input_layer.get_shape().ndims == 4)
    w_name = None if name is None else name + '_w'
    b_name = None if name is None else name + '_b'
    w = weight_variable([window, window, input_layer.get_shape().as_list()[-1], depth], w_name)
    b = bias_variable([depth], b_name)
    conv = tf.nn.conv2d(input_layer, w, strides=[1, 1, 1, 1], padding='SAME') + b
    output = tf.nn.sigmoid(conv)
    if pool is not None:
        output = tf.nn.max_pool(output, ksize=[1, pool, pool, 1], strides=[1, pool, pool, 1], padding='SAME')
    if variables is not None:
        variables['conv_w'].append(w)
        variables['conv_b'].append(b)
    return output

def ff_layer(input_layer, depth, name=None, activation=True, variables=None):
    """Construct a fully connected layer which takes input_layer as input.

    input_layer -> output
    (batch_size, input_depth) -> (batch_size, depth)

    :param input_layer:
    :param depth: number of output nodes
    :param name:
    :param activation: boolean for whether to use the activation function (should be False for last layer)
    :param variables: dict with keys ff_w and ff_b to add weight and bias variables to
    """
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

def conv_to_ff_layer(input_layer):
    """Collapse a convolutional layer into a single dimension (plus batch dimension).

    input_layer -> output
    (batch_size, height, width, input_depth) -> (batch_size, height*width*input_depth)

    :param input_layer:
    """
    shape = input_layer.get_shape().as_list()
    output = tf.reshape(input_layer, [shape[0], reduce(operator.mul, shape[1:], 1)])
    return output

def model(data):
    """Construct a model.

    :param data: the batched input images
    """
    variables = defaultdict(list)
    conv = conv_layer(data, depth=64, window=5, pool=2, name='conv1', variables=variables)
    conv = conv_layer(conv, depth=32, window=5, pool=2, name='conv2', variables=variables)
    reshape = conv_to_ff_layer(conv)
    hidden = ff_layer(reshape, depth=512, name='ff1', variables=variables)
    output = ff_layer(hidden, depth=NUM_LABELS, name='ff2', activation=False, variables=variables)
    return output, variables

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--description", type=str, default="No description Provided", help="A helpful label for this run")
    parser.add_argument("-s", "--checkpoint-frequency", type=int, default=200, help="Frequency of saving the state of training (in steps)")
    parser.add_argument("-k", "--checkpoint-max-keep", type=int, default=5, help="The maximum number of checkpoints to keep before deleting old ones")
    parser.add_argument("-t", "--checkpoint-hours", type=int, default=6, help="Always keep 1 checkpoint every n hours")
    parser.add_argument("-f", "--load-file", type=str, help="filename of saved checkpoint")
    parser.add_argument("-o", "--checkpoint-label", type=str, default="checkpoint__"+hash+"__"+timestamp,help="Saved checkpoints will be named 'name'-'step'. defaults to checkpoint__hash__timestamp")

    args = parser.parse_args()

    print("Starting Training......Git commit : %s\n Model: TODO\n"%hash)
    print("Description:")
    print("\t"+args.description+"\n")
    print("Saving every %d steps, keeping a maximum of %d old checkpoints but keeping one checkpoint every %d hours." % (args.checkpoint_frequency, args.checkpoint_max_keep, args.checkpoint_hours))
   
    if not os.path.exists(CHECKPOINT_DIRECTORY):
        print("Directory for checkpoints doesn't exist! Creating directory '" + CHECKPOINT_DIRECTORY +     "/'")
        os.makedirs(CHECKPOINT_DIRECTORY)
    else:
        print("Checkpoints will be saved to '%s'" %(CHECKPOINT_DIRECTORY+"/"))    

 
    x = tf.placeholder(TYPE, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name='input')
    y = tf.placeholder(tf.int32, shape=(BATCH_SIZE,), name='labels')

    cats = ['abbey', 'playground']
    train_data, train_labels, train_files = get_files('train', cats, n=IMAGES_PER_CAT)
    train_size = len(train_files)
    val_data, val_labels, val_files = get_files('val', cats)

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

    batch_val_data, batch_val_labels = tf.train.batch(
            [val_data, val_labels],
            batch_size=BATCH_SIZE)

    start_time = time.time()
    config = tf.ConfigProto()
    # config.operation_timeout_in_ms = 2000
    
    saver = tf.train.Saver(max_to_keep = args.checkpoint_max_keep, keep_checkpoint_every_n_hours = args.checkpoint_hours)

    with tf.Session(config=config) as sess:
        # Run all the initializers to prepare the trainable parameters.
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print('Initialized!')

        val_data_sample, val_labels_sample = sess.run([batch_val_data, batch_val_labels])
        val_feed_dict = {x: val_data_sample, y: val_labels_sample}

        # unsure this is the right place to restore variable states
        if args.load_file:
            saver.restore(sess, args.load_file)
            print("Restored state from file : " + args.load_file)

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
                val_l, val_predictions = sess.run([loss, train_prediction],
                        feed_dict=val_feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                        (step, float(step) * BATCH_SIZE / train_size,
                            1000 * elapsed_time / EVAL_FREQUENCY))
                print('\tMinibatch loss: %.3f, regularizers: %.6g learning rate: %.6f' % (l, r, lr))
                print('\tMinibatch accuracy: %.1f%%' % (100 * accuracy(predictions, _labels)))
                print('\tValidation loss: %.3f Validation accuracy: %.1f%% \n' % (val_l, 100 * accuracy(val_predictions, val_labels_sample)))
                sys.stdout.flush()
            

            if step % args.checkpoint_frequency == 0:
                print("\tSaving state to %s......"% (CHECKPOINT_DIRECTORY+"/"+args.checkpoint_label+"-"+str(step)))
                saver.save(sess, CHECKPOINT_DIRECTORY+"/"+args.checkpoint_label, global_step = step)
                print("\tSuccess!\n")

        coord.request_stop()
        coord.join(threads)
