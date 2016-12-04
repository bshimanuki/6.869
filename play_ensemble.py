from collections import defaultdict
import sys
import time

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from constants import *
from util import *


def load_model(checkpoint_file, compute='logits'):
    meta = checkpoint_file[:checkpoint_file.rfind('-')] + '.meta'
    reader = tf.train.NewCheckpointReader(checkpoint_file)
    variables = reader.get_variable_to_shape_map().keys()
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session(graph=g)
        tf.train.import_meta_graph(meta)

        for key in variables:
            sess.run(sess.graph.get_collection(tf.GraphKeys.VARIABLES, '%s:0' % key)[-1].assign(reader.get_tensor(key)))
            # print("tensor_name: ", key)
            # print(reader.get_tensor(key))

        output = g.get_collection(compute)[-1]
        x = g.get_tensor_by_name('input:0')
        try:
            keep_prob = g.get_tensor_by_name('keep_prob:0')
        except KeyError:
            keep_prob = None

        def run(data):
            feed_dict = {x: data}
            if keep_prob is not None:
                feed_dict[keep_prob] = 1.
            return sess.run(output, feed_dict=feed_dict)
    return run, tf.placeholder(output.dtype, output.get_shape().as_list())


def run_model(values, placeholders, num_images=None, partition='test', shuffle=False):
    test_data, test_labels, test_files = get_input(partition, shuffle=shuffle)
    if num_images is None:
        num_images = get_size(partition)
    batch_data, batch_labels, batch_files = tf.train.batch(
        [test_data, test_labels, test_files],
        batch_size=BATCH_SIZE,
        capacity=5*BATCH_SIZE)

    y = tf.placeholder(tf.int32, shape=(None,), name='labels')
    accuracy_1 = 100 * accuracy(values, y)
    accuracy_5 = 100 * accuracy(values, y, k=5)
    accuracy_1_sum = 0.
    accuracy_5_sum = 0.

    _values = np.zeros((num_images, NUM_LABELS), dtype=np.float32)
    files = []

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        start_time = time.time()

        for step in range(1, num_images // BATCH_SIZE + 1):
            _data, _labels, _files = sess.run([batch_data, batch_labels, batch_files])

            test_feed_dict = {}
            for placeholder, func in placeholders.items():
                test_feed_dict[placeholder] = func(_data)
            test_feed_dict[y] = _labels

            _value, _accuracy_1, _accuracy_5 = sess.run([values, accuracy_1, accuracy_5], feed_dict=test_feed_dict)
            _values[(step-1)*BATCH_SIZE:step*BATCH_SIZE] = _value
            files.extend(_files)

            elapsed_time = time.time() - start_time
            start_time = time.time()
            accuracy_1_sum += _accuracy_1
            accuracy_5_sum += _accuracy_5
            if partition != 'test':
                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * BATCH_SIZE / num_images, 1000 * elapsed_time))
                print('\tMinibatch top-1 accuracy: %.1f%%, Minibatch top-5 accuracy: %.1f%%' %
                      (_accuracy_1, _accuracy_5))
                print('\tTotal top-1 accuracy: %.1f%%, Total top-5 accuracy: %.1f%%' %
                      (accuracy_1_sum/step, accuracy_5_sum/step))
                sys.stdout.flush()

        coord.request_stop()
        coord.join(threads)
        return _values, files


def show_image(files, predictions, i, k=10):
    predictions = enumerate(predictions[i])
    top_k = sorted(predictions, key=lambda x:x[1], reverse=True)[:k]
    for cat, prob in top_k:
        print('%s: %.6f' % (ALL_CATEGORIES[cat], prob))
    img = Image.open(files[i])
    img.resize((3*IMAGE_IMPORT_SIZE, 3*IMAGE_IMPORT_SIZE), Image.ANTIALIAS).show()


if __name__ == '__main__':
    # example, replace with correct checkpoints
    # should be of the form 'run-step' and 'run.meta'
    d = {}
    alex, ap = load_model('checkpoints/AlexNetSmall/2016-11-29_04:09:46__efec3a2-99000')
    d[ap] = alex
    vgg, vp = load_model('checkpoints/VGGNet/test-10000')
    d[vp] = vgg

    score = (tf.nn.softmax(ap) + tf.nn.softmax(vp)) / 2

    print('Graphs initialized.')
    predictions, files = run_model(score, d, partition='val', num_images=300)
    show_image(files, predictions, 0)
