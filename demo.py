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
from nn_util import num_parameters, layer_to_image_summary, weight_to_image_summary
from util import accuracy, get_input, get_size

from alexnet_small import AlexNetSmall
from briannet import BrianNet
from vggnet import VGGNet

timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")

# print to save to full log and console
full_log = logger.createLogger(LOGS_DIR + "full_output__" + timestamp + ".log", "all outputs", True)
print = full_log.info
# short git hash
githashval = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode('UTF-8')[:7]

# call log.info to save to proper log
log = logger.createLogger(LOGS_DIR + "log__" + timestamp + ".log", "logs to keep", False)
param_log_name = LOGS_DIR + "save__" + timestamp + "__" + githashval + ".log"
# param_log = logger.createLogger(param_log_name, "parameters", True)

import tensorflow as tf

def run(target_categories, optimizer, val_feed_dict_supp, train_feed_dict_supp, model, restore=False):
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--description", type=str, default="No description Provided", help="A helpful label for this run")
    parser.add_argument("-s", "--checkpoint-frequency", type=int, default=2000, help="Frequency of saving the state of training (in steps)")
    parser.add_argument("-k", "--checkpoint-max-keep", type=int, default=30, help="The maximum number of checkpoints to keep before deleting old ones")
    parser.add_argument("-t", "--checkpoint-hours", type=int, default=2, help="Always keep 1 checkpoint every n hours")
    parser.add_argument("-f", "--load-file", type=str, help="filename of saved checkpoint")
    parser.add_argument("-o", "--name", type=str, default='',help="Saved checkpoints will be named 'name'__'timestamp'-'step'. defaults to the git hash")

    args = parser.parse_args()
    if args.name:
        args.name += '__'
    args.name += githashval + '__' + timestamp

    print("Starting Training......Git commit : %s\n Model: TODO\n"%githashval)
    print("Description:")
    print("\t"+args.description+"\n")
    checkpoint_dir = CHECKPOINT_DIRECTORY + model.name() + '/'
    if args.checkpoint_frequency:
        print("Saving every %d steps, keeping a maximum of %d old checkpoints but keeping one checkpoint every %d hours." % (args.checkpoint_frequency, args.checkpoint_max_keep, args.checkpoint_hours))
        if not os.path.exists(checkpoint_dir):
            print("Directory for checkpoints doesn't exist! Creating directory '%s'" % checkpoint_dir)
            os.makedirs(checkpoint_dir)
        else:
            print("Checkpoints will be saved to '%s'" % checkpoint_dir)
    else:
        print("Not saving checkpoints.")

    checkpoint_prefix = checkpoint_dir + args.name
    tensorboard_prefix = TB_LOGS_DIR + model.name() + '/' + args.name + '/'

    x = tf.placeholder(TYPE, shape=(None, IMAGE_FINAL_SIZE, IMAGE_FINAL_SIZE, NUM_CHANNELS), name='input')
    y = tf.placeholder(tf.int32, shape=(None,), name='labels')

    print('Using %d categories.' % (len(set(target_categories) & set(ALL_CATEGORIES)) if target_categories else len(ALL_CATEGORIES)))

    train_data, train_labels, _ = get_input('train', target_categories, n=IMAGES_PER_CAT)
    train_size = get_size('train', target_categories, n=IMAGES_PER_CAT)
    val_data, val_labels, _ = get_input('val', target_categories)

    logits, variables = model.model(x)
    prediction = tf.nn.softmax(logits)
    softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))

    global_step = tf.Variable(0, trainable=False)
    conv_reg_k = tf.train.exponential_decay(1e-4, global_step, decay_steps=10000, decay_rate=0.1, staircase=False)
    ff_reg_k = 1e-4

    conv_regularizers = sum(map(tf.nn.l2_loss, variables['conv_w'] + variables['conv_b']))
    ff_regularizers = sum(map(tf.nn.l2_loss, variables['ff_w'] + variables['ff_b']))
    conv_reg_loss = conv_reg_k * conv_regularizers
    ff_reg_loss = ff_reg_k * ff_regularizers
    loss = softmax_loss + conv_reg_loss + ff_reg_loss

    loss_summary = tf.scalar_summary('Loss', softmax_loss)
    conv_reg_loss_summary = tf.scalar_summary('Convolution Regularizer Loss', conv_reg_loss)
    ff_reg_loss_summary = tf.scalar_summary('Fully Connected Regularizer Loss', ff_reg_loss)
    total_loss_summary = tf.scalar_summary('Total Loss', loss)
    extra_loss_summaries = tf.merge_summary([conv_reg_loss_summary, ff_reg_loss_summary, total_loss_summary])

    optimizer_op = optimizer.minimize(loss, global_step=global_step)

    print('Model %s has %d parameters.' % (model.name(), num_parameters(variables)))
    print('\t%d conv parameters.' % num_parameters({k:v for k,v in variables.items() if k.startswith('conv')}))
    print('\t%d ff parameters.' % num_parameters({k:v for k,v in variables.items() if k.startswith('ff')}))

    tf.add_to_collection('logits', logits)
    tf.add_to_collection('prediction', prediction)

    with tf.name_scope('top1'):
        accuracy_1 = 100 * accuracy(logits, y)
        accuracy_1_summary = tf.scalar_summary('Top 1 Accuracy', accuracy_1)
    with tf.name_scope('top5'):
        accuracy_5 = 100 * accuracy(logits, y, k=5)
        accuracy_5_summary = tf.scalar_summary('Top 5 Accuracy', accuracy_5)

    merged_summaries = tf.merge_all_summaries()
    metric_summaries = tf.merge_summary([loss_summary, accuracy_1_summary, accuracy_5_summary])
    metric_with_loss_summaries = tf.merge_summary([metric_summaries, extra_loss_summaries])

    saver = tf.train.Saver(max_to_keep = args.checkpoint_max_keep, keep_checkpoint_every_n_hours = args.checkpoint_hours)
    if args.checkpoint_frequency:
        saver.export_meta_graph(checkpoint_prefix + '.meta')
        print('Model graph saved.')

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
        # Initialize summary writer for TensorBoard
        os.makedirs(tensorboard_prefix + 'training/')
        train_writer = tf.train.SummaryWriter(tensorboard_prefix + 'training/', graph=sess.graph)
        os.makedirs(tensorboard_prefix + 'validation/')
        val_writer = tf.train.SummaryWriter(tensorboard_prefix + 'validation/')

        # Run all the initializers to prepare the trainable parameters.
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print('Initialized!')

        # unsure this is the right place to restore variable states
        if args.load_file:
            saver.restore(sess, args.load_file)
            print("Restored state from file : " + args.load_file)

        # Loop through training steps.
        for step in range(1, NUM_EPOCHS * train_size // BATCH_SIZE + 1):

            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            _data, _labels = sess.run([batch_data, batch_labels])
            train_feed_dict = {x: _data, y: _labels}
            train_feed_dict.update(train_feed_dict_supp)

            # Run the optimizer to update weights can calculate loss
            if step % HISTOGRAM_FREQUENCY == 0:
                train_summaries = merged_summaries
            else:
                train_summaries = metric_with_loss_summaries
            _, train_l, train_predictions, minibatch_top1, minibatch_top5, train_summary = sess.run(
                    [optimizer_op, loss, prediction, accuracy_1, accuracy_5, train_summaries],
                    feed_dict=train_feed_dict)
            if step >= MIN_EVAL_STEP:
                train_writer.add_summary(train_summary, step)

            # print some extra information once reach the evaluation frequency
            if step % EVAL_FREQUENCY == 0 and step >= MIN_EVAL_STEP:
                # get next validation batch
                val_data_sample, val_labels_sample = sess.run([batch_val_data, batch_val_labels])
                val_feed_dict = {x: val_data_sample, y: val_labels_sample}
                val_feed_dict.update(val_feed_dict_supp)

                # calculate validation set metrics
                val_l, val_predictions, val_top1, val_top5, val_summary = sess.run(
                        [loss, prediction, accuracy_1, accuracy_5, metric_summaries],
                        feed_dict=val_feed_dict)

                # Add TensorBoard summary to summary writer
                val_writer.add_summary(val_summary, step)
                train_writer.flush()
                val_writer.flush()

                # Print info/stats
                elapsed_time = time.time() - start_time
                start_time = time.time()

                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * BATCH_SIZE / train_size,
                       1000 * elapsed_time / EVAL_FREQUENCY))
                print('\tMinibatch loss: %.3f' % (train_l))
                # print('\tLearning rate: %.6f' % (learning_rate))
                # TODO: we don't actually have a reliable way of determining the learning rate right now
                print('\tMinibatch top-1 accuracy: %.1f%%, Minibatch top-5 accuracy: %.1f%%' %
                      (minibatch_top1, minibatch_top5))
                print('\tValidation loss: %.3f' % (val_l))
                print('\tValidation top-1 accuracy: %.1f%%, Validation top-5 accuracy: %.1f%%' %
                      (val_top1, val_top5))
                sys.stdout.flush()

            if variables['conv_w'] and (step == 1 or step % EVAL_IMAGE_FREQUENCY == 0):
                summary = weight_to_image_summary(variables['conv_w'][0], name='weights/%d'%step)
                _summary = sess.run(summary)
                train_writer.add_summary(_summary)
                train_writer.flush()
                print('Added image summary.')

            if args.checkpoint_frequency and step % args.checkpoint_frequency == 0:
                print("\tSaving state to %s......" % (checkpoint_prefix + "-" + str(step)))
                saver.save(sess, checkpoint_prefix, global_step = step, write_meta_graph=False)
                print("\tSuccess!\n")

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    # optimizer = tf.train.AdamOptimizer(0.001)
    optimizer = tf.train.AdagradOptimizer(0.01)
    target_categories = []
    # target_categories = ['playground', 'abbey', 'amphitheater', 'baseball_field', 'bedroom', 'cemetery', 'courtyard', 'kitchen', 'mountain', 'shower']
    # target_categories = ALL_CATEGORIES[:10]

    keep_prob = tf.placeholder(tf.float32, name='keep_prob') # we need to define a probability for the dropout

    ### Example when running BrianNet
    #run(target_categories, optimizer, {}, {}, model=BrianNet())

    # model = AlexNetSmall(keep_prob)
    model = VGGNet(keep_prob)
    run(target_categories, optimizer, {keep_prob: 1.}, {keep_prob: KEEP_PROB}, model=model)
