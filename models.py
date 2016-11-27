from collections import defaultdict

import tensorflow as tf

from constants import NUM_LABELS
from nn_util import conv_layer, conv_to_ff_layer, ff_layer


def briannet(data):
    """Construct a model.

    :param data: the batched input images
    """
    variables = defaultdict(list)
    conv = conv_layer(data, depth=64, stride=1, window=5, pool=(2, 2), name='conv1', variables=variables)
    conv = conv_layer(conv, depth=32, stride=1, window=5, pool=(2, 2), name='conv2', variables=variables)
    reshape = conv_to_ff_layer(conv)
    hidden = ff_layer(reshape, depth=512, name='ff1', variables=variables)
    output = ff_layer(hidden, depth=NUM_LABELS, name='ff2', activation=False, variables=variables)
    return output, variables


def alexnet(data, keep_prob):
    variables = defaultdict(list)
    conv1 = conv_layer(
        data,
        depth=96,
        stride=4,
        window=11,
        activation_fn=tf.nn.relu,
        pool=(3, 2),
        lrn=(5, 1.0, 1e-4, 0.75),
        name='conv1',
        variables=variables
    )
    conv2 = conv_layer(
        conv1,
        depth=256,
        stride=1,
        window=5,
        activation_fn=tf.nn.relu,
        pool=(3, 2),
        lrn=(5, 1.0, 1e-4, 0.75),
        name='conv2',
        variables=variables
    )
    conv3 = conv_layer(
        conv2,
        depth=384,
        stride=1,
        window=3,
        activation_fn=tf.nn.relu,
        name='conv3',
        variables=variables
    )
    conv4 = conv_layer(
        conv3,
        depth=256,
        stride=1,
        window=3,
        activation_fn=tf.nn.relu,
        name='conv4',
        variables=variables
    )
    conv5 = conv_layer(
        conv4,
        depth=256,
        stride=1,
        window=3,
        activation_fn=tf.nn.relu,
        pool=(3, 2),
        name='conv5',
        variables=variables
    )
    conv5r = conv_to_ff_layer(conv5)
    # ff_layer(input_layer, depth, activation_fn=tf.nn.sigmoid, dropout=None, name=None, activation=True, variables=None):
    fc6 = ff_layer(
        conv5r,
        depth=4096,
        activation_fn = tf.nn.relu,
        dropout = keep_prob,
        name='fc6',
        variables=variables
    )
    fc7 = ff_layer(
        fc6,
        depth=4096,
        activation_fn = tf.nn.relu,
        dropout = keep_prob,
        name='fc7',
        variables=variables
    )
    output = ff_layer(fc7, depth=NUM_LABELS, name='output', activation=False, variables=variables)
    return output, variables