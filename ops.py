import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm


def conv2d_cus(input_, output_dim, k_size=3, stride=2, stddev=0.02, name="conv2d", zero_init=False):
    with tf.variable_scope(name):
        if zero_init:
            initializer = tf.zeros_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)
        w = tf.get_variable('w', [k_size, k_size, input_.get_shape()[-1], output_dim],
                            initializer=initializer)
        conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        return conv


# modified code from https://github.com/openai/improved-gan/blob/master/imagenet/ops.py
def conv_transpose(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                   name="deconv2d", with_w=False, init_bias=0):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(init_bias))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


# modified code from https://github.com/wenxinxu/resnet-in-tensorflow/blob/master/resnet.py
def residual_block(input_layer, output_channel, first_block=False, block_num=None,
                   avg_pool_add=True, is_training=True, zero_init=False, core=False):
    """
    Defines a residual block in ResNet
    :param zero_init: init weights for convolution as zeros
    :param core: add a core prefix in scope
    :param is_training: if network is currently training
    :param avg_pool_add: boolean to determine if the function do a residual avg pool addition if the dimension is changed
    :param block_num: number of the block (used for naming scope)
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    """

    block_num = "_" + x_str(block_num)
    n, h, w, input_channel = input_layer.get_shape().as_list()

    if core:
        scope_name = "core_" + "block" + block_num
    else:
        scope_name = "block" + block_num

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        # When it's time to "shrink" the image size, we use stride =
        # output channel can be >= input channel since the first block would go from 3 to 64
        if input_channel * 2 <= output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

        # The first conv layer of the first residual block does not need to be normalized and relu-ed.
        with tf.variable_scope('conv1_in_block' + block_num):
            if first_block:
                std_dev = tf.sqrt(2 / tf.reduce_prod(tf.cast(input_layer.get_shape()[1:], dtype=tf.float32)))
                conv1 = conv2d_cus(input_layer, output_channel, name='conv2d_1', stride=stride, k_size=3,
                                   stddev=std_dev, zero_init=zero_init)
            else:
                std_dev = tf.sqrt(2 / tf.reduce_prod(tf.cast(np.array([h, w, output_channel]), dtype=tf.float32)))
                conv1 = tf.nn.relu(batch_norm(conv2d_cus(input_layer, output_channel, name='conv2d_1', stddev=std_dev,
                                                         k_size=3, stride=stride, zero_init=zero_init),
                                              is_training=is_training, scope="bn1"))

        with tf.variable_scope('conv2_in_block' + block_num):
            std_dev = tf.sqrt(2 / tf.reduce_prod(tf.cast(np.array([h, w, output_channel]), dtype=tf.float32)))
            conv2 = tf.nn.relu(batch_norm(conv2d_cus(conv1, output_channel, name='conv2d_2', stddev=std_dev,
                                                     k_size=3, stride=1, zero_init=zero_init),
                                          is_training=is_training, scope="bn2"))

        if first_block or (not avg_pool_add and increase_dim):
            return conv2

        # When the channels of input layer and conv2 does not match, we add zero pads to increase the
        #  depth of input layers
        if increase_dim:
            pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            residual_orig_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                                 input_channel // 2]])
        else:
            residual_orig_input = input_layer

        output = conv2 + residual_orig_input
        return output


def x_str(s):
    if s is None:
        return ''
    return str(s)


def dense(x, output_features, name="dense"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        return tf.layers.dense(x, output_features)
