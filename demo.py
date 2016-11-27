#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import time

import logger
# all prints
from constants import *
from models import alexnet
from util import accuracy, get_categories, get_files

timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")

# print to save to full log and console
full_log = logger.createLogger(LOGS_DIR + "full_output__" + timestamp + ".log", "all outputs", True)
print = full_log.info
githashval = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode('UTF-8')[:-1]

# call log.info to save to proper log
log = logger.createLogger(LOGS_DIR + "log__" + timestamp + ".log", "logs to keep", False)
param_log_name = LOGS_DIR + "save" + githashval + "__" + timestamp +".log"
# param_log = logger.createLogger(param_log_name, "parameters", True)

import tensorflow as tf


def run(cats, learning_rate, optimizer, val_feed_dict_supp, train_feed_dict_supp, model):
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--description", type=str, default="No description Provided", help="A helpful label for this run")
    parser.add_argument("-s", "--checkpoint-frequency", type=int, default=200, help="Frequency of saving the state of training (in steps)")
    parser.add_argument("-k", "--checkpoint-max-keep", type=int, default=10, help="The maximum number of checkpoints to keep before deleting old ones")
    parser.add_argument("-t", "--checkpoint-hours", type=int, default=6, help="Always keep 1 checkpoint every n hours")
    parser.add_argument("-f", "--load-file", type=str, help="filename of saved checkpoint")
    parser.add_argument("-o", "--checkpoint-label", type=str, default="checkpoint__"+githashval+"__"+timestamp,help="Saved checkpoints will be named 'name'-'step'. defaults to checkpoint__hash__timestamp")

    args = parser.parse_args()

    print("Starting Training......Git commit : %s\n Model: TODO\n"%githashval)
    print("Description:")
    print("\t"+args.description+"\n")
    if args.checkpoint_frequency:
        print("Saving every %d steps, keeping a maximum of %d old checkpoints but keeping one checkpoint every %d hours." % (args.checkpoint_frequency, args.checkpoint_max_keep, args.checkpoint_hours))
        if not os.path.exists(CHECKPOINT_DIRECTORY):
            print("Directory for checkpoints doesn't exist! Creating directory '%s'" % CHECKPOINT_DIRECTORY)
            os.makedirs(CHECKPOINT_DIRECTORY)
        else:
            print("Checkpoints will be saved to '%s'" % CHECKPOINT_DIRECTORY)
    else:
        print("Not saving checkpoints.")

    (all_categories, category_to_index) = get_categories()

    x = tf.placeholder(TYPE, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name='input')
    y = tf.placeholder(tf.int32, shape=(BATCH_SIZE,), name='labels')

    train_data, train_labels, train_files = get_files('train', all_categories, cats, n=IMAGES_PER_CAT)
    train_size = len(train_files)
    val_data, val_labels, val_files = get_files('val', all_categories, cats)

    logits, variables = model(x)
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

    batch_data, batch_labels = tf.train.batch(
            [train_data, train_labels],
            batch_size=BATCH_SIZE)

    batch_val_data, batch_val_labels = tf.train.batch(
            [val_data, val_labels],
            batch_size=BATCH_SIZE)

    start_time = time.time()
    if USE_GPU:
        config = tf.ConfigProto()
    else:
        config = tf.ConfigProto(device_count={'GPU': 0})
    # config.operation_timeout_in_ms = 2000

    saver = tf.train.Saver(max_to_keep = args.checkpoint_max_keep, keep_checkpoint_every_n_hours = args.checkpoint_hours)

    with tf.Session(config=config) as sess:
        # Initialize summary writer for TensorBoard
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(TB_LOGS_DIR + 'training/', graph=sess.graph)
        val_writer = tf.train.SummaryWriter(TB_LOGS_DIR + 'validation/')

        # Run all the initializers to prepare the trainable parameters.
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print('Initialized!')

        val_data_sample, val_labels_sample = sess.run([batch_val_data, batch_val_labels])
        val_feed_dict = {x: val_data_sample, y: val_labels_sample}
        val_feed_dict.update(val_feed_dict_supp)

        # unsure this is the right place to restore variable states
        if args.load_file:
            saver.restore(sess, args.load_file)
            print("Restored state from file : " + args.load_file)

        # Loop through training steps.
        for step in range(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
            _data, _labels = sess.run([batch_data, batch_labels])

            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            train_feed_dict = {x: _data, y: _labels}
            train_feed_dict.update(train_feed_dict_supp)
            # Run the optimizer to update weights.
            sess.run(optimizer_op, feed_dict=train_feed_dict)

            # print some extra information once reach the evaluation frequency
            if step % EVAL_FREQUENCY == 0 and step > 0:
                # fetch some extra nodes' data
                train_l, train_predictions, minibatch_top1, minibatch_top5, train_summary = sess.run(
                        [loss, prediction, accuracy_1, accuracy_5, merged],
                        feed_dict=train_feed_dict) # TODO: correct feed dict for summary?

                # calculate validation set metrics
                val_l, val_predictions, val_top1, val_top5, val_summary = sess.run(
                        [loss, prediction, accuracy_1, accuracy_5, metric_summaries],
                        feed_dict=val_feed_dict)

                # Add TensorBoard summary to summary writer
                train_writer.add_summary(train_summary, step)
                train_writer.flush()
                val_writer.add_summary(val_summary, step)
                val_writer.flush()

                # Print info/stats
                elapsed_time = time.time() - start_time
                start_time = time.time()

                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * BATCH_SIZE / train_size,
                       1000 * elapsed_time / EVAL_FREQUENCY))
                print('\tMinibatch loss: %.3f' % (train_l))
                print('\tLearning rate: %.6f' % (learning_rate))
                print('\tMinibatch top-1 accuracy: %.1f%%, Minibatch top-5 accuracy: %.1f%%' %
                      (minibatch_top1, minibatch_top5))
                print('\tValidation loss: %.3f' % (val_l))
                print('\tValidation top-1 accuracy: %.1f%%, Validation top-5 accuracy: %.1f%%' %
                      (val_top1, val_top5))
                sys.stdout.flush()


            if args.checkpoint_frequency and step % args.checkpoint_frequency == 0:
                print("\tSaving state to %s......" % (
                CHECKPOINT_DIRECTORY + args.checkpoint_label + "-" + str(step)))
                saver.save(sess, CHECKPOINT_DIRECTORY + args.checkpoint_label, global_step = step)
                print("\tSuccess!\n")

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    learning_rate = 0.002
    """
    TODO: I need to figure out how to print out the learning rate when it's not a constant
    Currently the issue is that the variable learning rate (using tf.train.exponential_decay) gives a float, but
    setting tf.Variable(0.002) give a list (!!)
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    ### Example when running BrianNet
    # run([], 0.002, optimizer, {}, {}, briannet)

    ### Example when running AlexNet
    keep_prob = tf.placeholder(tf.float32, name='keep_prob') # we need to define a probability for the dropout
    run([], 0.002, optimizer, {keep_prob: 1.}, {keep_prob: KEEP_PROB}, model = lambda v: alexnet(v, keep_prob))
