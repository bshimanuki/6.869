from collections import defaultdict

import tensorflow as tf

from constants import NUM_LABELS
from nn_util import conv_layer, conv_to_ff_layer, ff_layer
from model import Model

class BrianNet(Model):
    def model(self, data):
        """Construct a model.

        :param data: the batched input images
        """
        variables = defaultdict(list)
        conv = conv_layer(data, depth=64, window=5, pool=(2, 2), name='conv1', variables=variables)
        conv = conv_layer(conv, depth=32, window=5, pool=(2, 2), name='conv2', variables=variables)
        reshape = conv_to_ff_layer(conv)
        hidden = ff_layer(
            reshape, 
            depth=512, 
            name='ff1', 
            variables=variables
        )
        output = ff_layer(
            hidden, 
            depth=NUM_LABELS, 
            name='ff2', 
            activation=False, 
            variables=variables
        )
        return output, variables
