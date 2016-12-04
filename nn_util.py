import math
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
    with tf.name_scope('weight'):
        weight = tf.Variable(tf.truncated_normal(
            shape,
            stddev=sigma,
            seed=SEED,
            dtype=TYPE))
        tf.histogram_summary('%s/weight' % (name if name is not None else ''), weight)
        return weight


def bias_variable(shape, name=None):
    with tf.name_scope('bias'):
        bias = tf.Variable(tf.zeros(shape=shape, dtype=TYPE))
        tf.histogram_summary('%s/bias' % (name if name is not None else ''), bias)
        return bias


def num_parameters(variables):
    n = 0
    for variables_by_type in variables.values():
        for variable in variables_by_type:
            n += reduce(operator.mul, variable.get_shape().as_list())
    return n


def conv_layer(input_layer, depth, window, stride=1, activation_fn=tf.nn.relu, pool=None, lrn=None, name=None,
               variables=None):
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
    with tf.name_scope(name):
        assert(input_layer.get_shape().ndims == 4)
        w = weight_variable([window, window, input_layer.get_shape().as_list()[-1], depth], name)
        b = bias_variable([depth], name)
        conv = tf.nn.conv2d(input_layer, w, strides=[1, stride, stride, 1], padding='SAME') + b
        # Note: Sample code seems to use tf.nn.bias_add instead of straight addition here.
        with tf.name_scope('output/' + name):
            output = activation_fn(conv, name='activation')
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
        tf.histogram_summary('%s/activation' % (name if name is not None else ''), output)
        tf.add_to_collection(name, output)
        return output


def ff_layer(input_layer, depth, activation_fn=tf.nn.relu, dropout=None, name=None, activation=True, variables=None):
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
    with tf.name_scope(name):
        assert(input_layer.get_shape().ndims == 2)
        w = weight_variable([input_layer.get_shape().as_list()[-1], depth], name)
        b = bias_variable([depth], name)
        with tf.name_scope('hidden/' + name):
            hidden = tf.matmul(input_layer, w) + b
            if activation:
                # TODO: potentially change this to just passign in an identity as the activation function
                hidden = activation_fn(hidden, name='activation')
            if dropout is not None:
                keep_prob = dropout
                hidden = tf.nn.dropout(hidden, keep_prob)
        if variables is not None:
            variables['ff_w'].append(w)
            variables['ff_b'].append(b)
        tf.histogram_summary('%s/hidden' % (name if name is not None else ''), hidden)
        tf.add_to_collection(name, hidden)
        return hidden


def conv_to_ff_layer(input_layer):
    """Collapse a convolutional layer into a single dimension (plus batch dimension).

    input_layer -> output
    (batch_size, height, width, input_depth) -> (batch_size, height*width*input_depth)

    :param input_layer:
    """
    with tf.name_scope('conv_to_ff_layer'):
        shape = input_layer.get_shape().as_list()
        output = tf.reshape(input_layer, [-1, reduce(operator.mul, shape[1:], 1)])
        return output


def layer_to_image_summary(layer, name=None):
    """Use for convolutional layers.
    Show the activations of an input image.
    """
    with tf.name_scope('layer_summary'):
        # get the first image
        v = layer
        batch_size, iy, ix, depth = v.get_shape().as_list()
        v = tf.slice(layer, (0,0,0,0), (1,-1,-1,-1))
        v = tf.reshape(v, (iy, ix, depth))
        cy = math.ceil(math.sqrt(depth))
        cx = cy
        v = tf.pad(v, ((0,0), (0,0), (0,cy*cx-depth)))
        v = tf.reshape(v, (iy, ix, cy, cx))
        v = tf.transpose(v, (2,0,3,1)) # cy,iy,cx,ix
        v = tf.reshape(v, (1, cy*iy, cx*ix, 1))
        return tf.image_summary(name, v)

def weight_to_image_summary(weight, name=None, max_images=1):
    """Use for first convolutional layer."""
    with tf.name_scope('weight_summary'):
        v = weight
        iy, ix, channels, depth = v.get_shape().as_list()
        cy = math.ceil(math.sqrt(depth))
        cx = cy
        v = tf.pad(v, ((0,0), (0,0), (0,0), (0,cy*cx-depth)))
        v = tf.reshape(v, (iy, ix, channels, cy, cx))
        v = tf.transpose(v, (3,0,4,1,2)) # cy,iy,cx,ix,channels
        v = tf.reshape(v, (1, cy*iy, cx*ix, channels))
        return tf.image_summary(name, v, max_images=max_images)
