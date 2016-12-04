from collections import defaultdict

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
    batch_data, batch_files = tf.train.batch(
        [test_data, test_files],
        batch_size=BATCH_SIZE,
        capacity=5*BATCH_SIZE)

    _values = np.zeros((num_images, NUM_LABELS), dtype=np.float32)
    files = []

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(num_images // BATCH_SIZE):
            _data, _files = sess.run([batch_data, batch_files])

            test_feed_dict = {}
            for placeholder, func in placeholders.items():
                test_feed_dict[placeholder] = func(_data)

            _value = sess.run(values, feed_dict=test_feed_dict)
            _values[step*BATCH_SIZE:(step+1)*BATCH_SIZE] = _value
            files.extend(_files)

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

    predictions, files = run_model(score, d, num_images=100)
    show_image(files, predictions, 0)
