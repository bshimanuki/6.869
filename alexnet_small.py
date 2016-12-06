from collections import defaultdict

import tensorflow as tf

from constants import NUM_LABELS
from nn_util import conv_layer, conv_to_ff_layer, ff_layer
from model import Model

class AlexNetSmall(Model):
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob

    def model(self, data):
        variables = defaultdict(list)
        conv1 = conv_layer(data, depth=96, window=6, stride=2, activation_fn=tf.nn.relu, pool=(3, 2),
                           lrn=(5, 1.0, 1e-4, 0.75), name='conv1', variables=variables)
        conv2 = conv_layer(conv1, depth=256, window=5, activation_fn=tf.nn.relu, pool=(3, 2),
                           lrn=(5, 1.0, 1e-4, 0.75), name='conv2', variables=variables)
        conv3 = conv_layer(conv2, depth=384, window=3, activation_fn=tf.nn.relu, name='conv3',
                           variables=variables)
        conv4 = conv_layer(conv3, depth=384, window=3, activation_fn=tf.nn.relu, name='conv4',
                           variables=variables)
        conv5 = conv_layer(conv4, depth=256, window=3, activation_fn=tf.nn.relu, pool=(3, 2), name='conv5',
                           variables=variables)
        conv5r = conv_to_ff_layer(conv5)
        # ff_layer(input_layer, depth, activation_fn=tf.nn.sigmoid, dropout=None, name=None, activation=True, variables=None):
        fc6 = ff_layer(
            conv5r,
            depth=1024, # TODO: return to 4096
            activation_fn = tf.nn.relu,
            dropout = self.keep_prob,
            name='fc6',
            variables=variables
        )
        fc7 = ff_layer(
            fc6,
            depth=512, # TODO: return to 4096
            activation_fn = tf.nn.relu,
            dropout = self.keep_prob,
            name='fc7',
            variables=variables
        )
        output = ff_layer(fc7, depth=NUM_LABELS, name='output', activation=False, variables=variables)
        return output, variables
