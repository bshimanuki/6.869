#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import time

import logger
# all prints
from constants import *
from model import Model
from nn_util import num_parameters
from util import accuracy, get_input, get_size

from alexnet_small import AlexNetSmall
from briannet import BrianNet
from vggnet import VGGNet

import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--description", type=str, default="No description Provided", help="A helpful label for this run")
    parser.add_argument("-s", "--checkpoint-frequency", type=int, default=2000, help="Frequency of saving the state of training (in steps)")
    parser.add_argument("-k", "--checkpoint-max-keep", type=int, default=30, help="The maximum number of checkpoints to keep before deleting old ones")
    parser.add_argument("-t", "--checkpoint-hours", type=int, default=2, help="Always keep 1 checkpoint every n hours")
    parser.add_argument("-f", "--load-file", type=str, help="filename of saved checkpoint")
    parser.add_argument("-o", "--name", type=str, default="ENSEMBLE_TEST",help="Saved checkpoints will be named 'name'-'step'. defaults to timestamp__hash")
    #TODO: CHANGE THIS 

    args = parser.parse_args()





    num_networks = 2
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    optimizer = tf.train.AdadeltaOptimizer()
    target_categories = ['playground', 'abbey']

    alexnet = AlexNetSmall(keep_prob)
    val_feed_dict_supp = {keep_prob: 1.}
    train_feed_dict_supp = {keep_prob: KEEP_PROB}


    x = tf.placeholder(TYPE, shape=(BATCH_SIZE, IMAGE_FINAL_SIZE, IMAGE_FINAL_SIZE, NUM_CHANNELS), name='input')
    y = tf.placeholder(tf.int32, shape=(BATCH_SIZE,), name='labels')

    train_data, train_labels = get_input('train', target_categories, n=IMAGES_PER_CAT)
    train_size = get_size('train', target_categories, n=IMAGES_PER_CAT)
    val_data, val_labels = get_input('val', target_categories)

    logits, variables = alexnet.model(x)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    loss_summary = tf.scalar_summary('Loss', loss)
    optimizer_op = optimizer.minimize(loss)

    prediction = tf.nn.softmax(logits)

    with tf.name_scope('top1'):
        accuracy_1 = 100 * accuracy(logits, y)
        accuracy_1_summary = tf.scalar_summary('Top 1 Accuracy', accuracy_1)
    with tf.name_scope('top5'):
        accuracy_5 = 100 * accuracy(logits, y, k=5)
        accuracy_5_summary = tf.scalar_summary('Top 5 Accuracy', accuracy_5)

    metric_summaries = tf.merge_summary([loss_summary, accuracy_1_summary, accuracy_5_summary])

    saver = tf.train.Saver(max_to_keep = args.checkpoint_max_keep, keep_checkpoint_every_n_hours = args.checkpoint_hours)

    batch_data, batch_labels = tf.train.batch(
            [train_data, train_labels],
            batch_size=BATCH_SIZE,
            capacity=5*BATCH_SIZE)

    batch_val_data, batch_val_labels = tf.train.batch(
            [val_data, val_labels],
            batch_size=BATCH_SIZE,
            capacity=5*BATCH_SIZE)

    start_time = time.time()
    if USE_GPU:
        config = tf.ConfigProto()
    else:
        config = tf.ConfigProto(device_count={'GPU': 0})
    # config.operation_timeout_in_ms = 2000

    with tf.Session(config=config) as sess:
        merged = tf.merge_all_summaries()
        os.makedirs(tensorboard_prefix + 'training/')
        train_writer = tf.train.SummaryWriter(tensorboard_prefix + 'training/', graph=sess.graph)
        os.makedirs(tensorboard_prefix + 'validation/')
        val_writer = tf.train.SummaryWriter(tensorboard_prefix + 'validation/')

        # Run all the initializers to prepare the trainable parameters.
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('restoring session')
        saver.restore(sess, 'checkpoints/alexnet')
        print('restored')
        _data, _labels = sess.run([batch_data, batch_labels])
        train_feed_dict = {x: _data, y: _labels}
        train_feed_dict.update(train_feed_dict_supp)
        print('test1')
        _, train_l, train_predictions, minibatch_top1, minibatch_top5, train_summary = sess.run(
                [optimizer_op, loss, prediction, accuracy_1, accuracy_5, merged],
                feed_dict=train_feed_dict)
        print('test2')
        val_data_sample, val_labels_sample = sess.run([batch_val_data, batch_val_labels])
        val_feed_dict = {x: val_data_sample, y: val_labels_sample}
        val_feed_dict.update(val_feed_dict_supp)

        print('calculating validation set metrics')
        # calculate validation set metrics
        val_l, val_predictions, val_top1, val_top5, val_summary = sess.run(
                [loss, prediction, accuracy_1, accuracy_5, metric_summaries],
                feed_dict=val_feed_dict)

        print(val_predictions);
        sess



# def ensemble_predictions():
#     keep_prob = tf.placeholder(tf.float32, name='keep_prob')
#     alexnet = AlexNetSmall(keep_prob)
#     vggnet = VGGNet(keep_prob)

#     # Empty list of predicted labels for each of the neural networks.
#     pred_labels = []

#     # Classification accuracy on the test-set for each network.
#     test_accuracies = []

#     # Classification accuracy on the validation-set for each network.
#     val_accuracies = []

#     # For each neural network in the ensemble.
#     for i in range(num_networks):
#         # Reload the variables into the TensorFlow graph.
#         saver.restore(sess=session, save_path=get_save_path(i))

#         # Calculate the classification accuracy on the test-set.
#         test_acc = test_accuracy()

#         # Append the classification accuracy to the list.
#         test_accuracies.append(test_acc)

#         # Calculate the classification accuracy on the validation-set.
#         val_acc = validation_accuracy()

#         # Append the classification accuracy to the list.
#         val_accuracies.append(val_acc)

#         # Print status message.
#         msg = "Network: {0}, Accuracy on Validation-Set: {1:.4f}, Test-Set: {2:.4f}"
#         print(msg.format(i, val_acc, test_acc))

#         # Calculate the predicted labels for the images in the test-set.
#         # This is already calculated in test_accuracy() above but
#         # it is re-calculated here to keep the code a bit simpler.
#         pred = predict_labels(images=data.test.images)

#         # Append the predicted labels to the list.
#         pred_labels.append(pred)
    
#     return np.array(pred_labels), \
#            np.array(test_accuracies), \
#            np.array(val_accuracies)
