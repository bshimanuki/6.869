from collections import defaultdict

import tensorflow as tf

from constants import NUM_LABELS
from nn_util import conv_layer, conv_to_ff_layer, ff_layer
from model import Model

class VGGNet(Model):
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob

    def model(self, data):
        variables = defaultdict(list)
        conv11 = conv_layer(data, depth=64, window=3,
                           name='conv11', variables=variables)
        conv12 = conv_layer(conv11, depth=64, window=3,
                           name='conv12', variables=variables, pool=(2,2))
        conv21 = conv_layer(conv12, depth=128, window=3,
                           name='conv21', variables=variables)
        conv22 = conv_layer(conv21, depth=128, window=3,
                           name='conv22', variables=variables, pool=(2,2))
        conv31 = conv_layer(conv22, depth=256, window=3,
                           name='conv31', variables=variables)
        conv32 = conv_layer(conv31, depth=256, window=3,
                           name='conv32', variables=variables)
        conv33 = conv_layer(conv32, depth=256, window=3,
                           name='conv33', variables=variables, pool=(2,2))
        conv41 = conv_layer(conv33, depth=512, window=3,
                           name='conv41', variables=variables)
        conv42 = conv_layer(conv41, depth=512, window=3,
                           name='conv42', variables=variables)
        conv43 = conv_layer(conv42, depth=512, window=3,
                           name='conv43', variables=variables, pool=(2,2))
        conv51 = conv_layer(conv43, depth=512, window=3,
                           name='conv51', variables=variables)
        conv52 = conv_layer(conv51, depth=512, window=3,
                           name='conv52', variables=variables)
        conv53 = conv_layer(conv52, depth=512, window=3,
                           name='conv53', variables=variables, pool=(2,2))
        conv5r = conv_to_ff_layer(conv53)
        # ff_layer(input_layer, depth, activation_fn=tf.nn.sigmoid, dropout=None, name=None, activation=True, variables=None):
        fc6 = ff_layer(
            conv5r,
            depth=1024,
            dropout = self.keep_prob,
            name='fc6',
            variables=variables
        )
        fc7 = ff_layer(
            fc6,
            depth=512,
            dropout = self.keep_prob,
            name='fc7',
            variables=variables
        )
        output = ff_layer(fc7, depth=NUM_LABELS, name='output', activation=False, variables=variables)
        return output, variables
