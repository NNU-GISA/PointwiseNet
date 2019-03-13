import tensorflow as tf
from utils.tf_util import _variable_on_cpu, conv2d
import numpy as np


def Top_K(index, top_k, batch_size, num_point):
    first = tf.constant(np.arange(batch_size).reshape(batch_size, 1))
    first = tf.tile(first, [1, num_point])
    first = tf.expand_dims(first, axis=-1)
    second = tf.constant(np.arange(num_point).reshape(num_point, 1))
    second = tf.tile(second, [batch_size, 1])
    second = tf.reshape(second, shape=[batch_size, -1])
    second = tf.expand_dims(second, axis=-1)

    concat = tf.concat([first, second], axis=2)
    concat = tf.tile(concat, [1, 1, top_k])
    concat = tf.reshape(concat, shape=[batch_size, num_point, top_k, -1])
    concat = tf.cast(concat, dtype=tf.int32)
    index = tf.cast(index[1], dtype=tf.int32)
    index = tf.reshape(index, shape=[batch_size, num_point, top_k, -1])
    fuse = tf.concat([concat, index], axis=3)

    return fuse


def VLAD(input_tensor, K, is_training, bn_decay, layer_name=None):
    '''
    :param input_tensor: shape=[32 x 1024 x 1 x 128]
    :param K: center number
    :param is_training: bool tensor
    :param bn_decay:
    :param layer_name: VLAD_layer
    :return:
        global_pool: shape=[32 x 1024]
    '''

    with tf.variable_scope(layer_name):
        batch_size = input_tensor.get_shape()[0].value
        num_point = input_tensor.get_shape()[1].value
        D = input_tensor.get_shape()[-1].value

        top_k = 3  # top k hightest score

        #--------------------- VLAD -----------------------------
        # l2_normal
        reshape = tf.reshape(input_tensor, shape=[batch_size, num_point, D])
        conv_norm = tf.nn.l2_normalize(reshape, dim=2)
        descriptor = tf.expand_dims(conv_norm, axis=-1, name='expanddim')  # descriptor is B x N x D x 1

        # convolution score
        conv = conv2d(descriptor, K, [1, D],
                      padding='VALID', stride=[1, 1],
                      bn=True, is_training=is_training,
                      scope='assignment', bn_decay=bn_decay,
                      activation_fn=None)
        # soft score
        a = tf.nn.softmax(conv)     # 32 x 1024 x 1 x K

        index = tf.nn.top_k(a, top_k)	#32 x 1024 x 1 x top_k
        # ----------------------Top_K ----------------------------
        # for each point find the best top_k centers'
        # index, corresponding to the top_k score
        select_center = Top_K(index, top_k, batch_size, num_point)
        #--------------------------------------------------------

        # VLAD center
        Center = _variable_on_cpu(name=layer_name+'_centers',
                                  shape=[D, K],
                                  initializer=tf.contrib.layers.xavier_initializer())

        # calculate the differences
        diff = tf.expand_dims(reshape, axis=-1) - Center

        weight_sum = diff * a
        weight_sum = tf.transpose(weight_sum, perm=[0, 1, 3, 2])
        weight_sum = tf.gather_nd(weight_sum, select_center)
        #---------------------------------------------------------

        net = conv2d(weight_sum, 128, [1, 1],
                     padding='VALID', stride=[1, 1],
                     bn=True, is_training=is_training,
                     scope='vlad_conv1', bn_decay=bn_decay)
        net = conv2d(net, 256, [1, 1],
                     padding='VALID', stride=[1, 1],
                     bn=True, is_training=is_training,
                     scope='vlad_conv2', bn_decay=bn_decay)

        # local pooling: merge the center feature
        pool1 = tf.nn.max_pool(net, ksize=[1, 1, top_k, 1], strides=[1,1,2,1], padding='VALID')
        concat = tf.concat([input_tensor, pool1], axis=-1)
        #---------------------------------------------------------

        net = conv2d(concat, 256, [1, 1],
                     padding='VALID', stride=[1, 1],
                     bn=True, is_training=is_training,
                     scope='vlad_conv3', bn_decay=bn_decay)
        net = conv2d(net, 256, [1, 1],
                     padding='VALID', stride=[1, 1],
                     bn=True, is_training=is_training,
                     scope='vlad_conv4', bn_decay=bn_decay)

        # global pooling
        # global_pool = tf.nn.max_pool(net, ksize=[1, num_point, 1, 1], strides=[1, 2, 2, 1], padding='VALID')
        # global_pool = tf.reshape(global_pool, [batch_size, -1])

        return net, index[1]


def VLAD_layer(input_tensor1,input_tensor2, K, is_training, bn_decay, layer_name=None):
    '''
    :param input_tensor: shape=[32 x 1024 x 1 x 256]
    :param K: center number
    :param is_training: bool tensor
    :param bn_decay:
    :param layer_name: VLAD_layer
    :return:
        global_pool: shape=[32 x 1024]
    '''

    with tf.variable_scope(layer_name):
        batch_size = input_tensor1.get_shape()[0].value
        num_point = input_tensor1.get_shape()[1].value
        D = input_tensor1.get_shape()[-1].value

        #--------------------- VLAD -----------------------------
        # l2_normal
        reshape = tf.reshape(input_tensor1, shape=[batch_size, num_point, D])
        conv_norm = tf.nn.l2_normalize(reshape, dim=2)
        descriptor = tf.expand_dims(conv_norm, axis=-1, name='expanddim')  # descriptor is B x N x D x 1

        # convolution score
        conv = conv2d(descriptor, K, [1, D],
                      padding='VALID', stride=[1, 1],
                      bn=True, is_training=is_training,
                      scope='assignment', bn_decay=bn_decay,
                      activation_fn=None)
        # soft score
        a = tf.nn.softmax(conv)     # 32 x 1024 x 1 x K

        # VLAD center
        Center = _variable_on_cpu(name=layer_name+'_centers',
                                  shape=[D, K],
                                  initializer=tf.contrib.layers.xavier_initializer())

        # calculate the differences
        diff = tf.expand_dims(reshape, axis=-1) - Center

        weight_sum = diff * a
        weight_sum = tf.transpose(weight_sum, perm=[0,1,3,2])

        pool1 = tf.nn.max_pool(weight_sum, ksize=[1, 1, K, 1], strides=[1, 1, 2, 1], padding='VALID')
        concat = tf.concat([input_tensor2, pool1], axis=-1)
        #---------------------------------------------------------

        net = conv2d(concat, 384, [1, 1],
                     padding='VALID', stride=[1, 1],
                     bn=True, is_training=is_training,
                     scope='vlad_conv1', bn_decay=bn_decay)
        net = conv2d(net, 1024, [1, 1],
                     padding='VALID', stride=[1, 1],
                     bn=True, is_training=is_training,
                     scope='vlad_conv2', bn_decay=bn_decay)

        # global pooling
        global_pool = tf.nn.max_pool(net, ksize=[1, num_point, 1, 1], strides=[1, 2, 2, 1], padding='VALID')
        global_pool = tf.reshape(global_pool, [batch_size, -1])

        return global_pool