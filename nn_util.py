import operator
from functools import reduce

import numpy as np
import tensorflow as tf

from constants import SEED, TYPE


def weight_variable(shape, name=None):
    """
    Standard deviation set using the weights suggested in the tutorial code at
    https://github.com/hangzhaomit/tensorflow-tutorial/tree/master/miniplaces

    :param shape:
    :param name:
    :return:
    """
    sigma=np.sqrt(2./np.product(shape[:-1]))
    return tf.Variable(
            tf.truncated_normal(
                shape,
                stddev=sigma,
                seed=SEED,
                dtype=TYPE),
            name=name)


def bias_variable(shape, name=None):
    return tf.Variable(tf.zeros(shape=shape, dtype=TYPE), name=name)


def conv_layer(input_layer, depth, stride, window, activation_fn=tf.nn.sigmoid, pool=None, lrn=None, name=None, variables=None):
    """Construct a convolutional layer which takes input_layer as input.

    input_layer -> output
    (batch_size, height, width, input_depth) -> (batch_size, height, width, depth)

    :param input_layer: input tensor
    :param depth: number of convolution images
    :param stride:
    :param window: size of convolutional kernel (side length)
    :param pool: None for no pooling. (ksize, stride) otherwise.
    :param lrn: None for no local response normalization. (depth radius, bias, alpha, beta) otherwise.
    :param name:
    :param variables: dict with keys conv_w and conv_b to add weight and bias variables to
    """
    assert(input_layer.get_shape().ndims == 4)
    w_name = None if name is None else name + '_w'
    b_name = None if name is None else name + '_b'
    w = weight_variable([window, window, input_layer.get_shape().as_list()[-1], depth], w_name)
    b = bias_variable([depth], b_name)
    conv = tf.nn.conv2d(input_layer, w, strides=[1, stride, stride, 1], padding='SAME') + b
    # Note: Sample code seems to use tf.nn.bias_add instead of straight addition here.
    output = activation_fn(conv)
    if lrn is not None:
        (lrn_depth_radius, lrn_bias, lrn_alpha, lrn_beta) = lrn
        output = tf.nn.local_response_normalization(
            output,
            depth_radius=lrn_depth_radius,
            bias=lrn_bias,
            alpha=lrn_alpha,
            beta=lrn_beta,
        )
    if pool is not None:
        (pool_ksize, pool_stride) = pool
        output = tf.nn.max_pool(output, ksize=[1, pool_ksize, pool_ksize, 1], strides=[1, pool_stride, pool_stride, 1], padding='SAME')
    if variables is not None:
        variables['conv_w'].append(w)
        variables['conv_b'].append(b)
    return output


def ff_layer(input_layer, depth, activation_fn=tf.nn.sigmoid, dropout=None, name=None, activation=True, variables=None):
    """Construct a fully connected layer which takes input_layer as input.

    input_layer -> output
    (batch_size, input_depth) -> (batch_size, depth)

    :param input_layer:
    :param depth: number of output nodes
    :param activation_fn:
    :param dropout: None if no dropout layer; keep_prob otherwise
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
        # TODO: potentially change this to just passign in an identity as the activation function
        hidden = activation_fn(hidden)
    if dropout is not None:
        keep_prob = dropout
        hidden = tf.nn.dropout(hidden, keep_prob)
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