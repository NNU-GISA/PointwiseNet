import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from KNN_search import KNN_search
from VLAD_layer import VLAD

def get_transform_K(inputs, is_training, bn_decay=None, K = 3):
    """ Transform Net, input is BxNx1xK gray image
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value

    net = tf_util.conv2d(inputs, 256, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv2', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32) + tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    #transform = tf_util.fully_connected(net, 3*K, activation_fn=None, scope='tfc3')
    transform = tf.reshape(transform, [batch_size, K, K])
    return transform


def get_transform(point_cloud, is_training, bn_decay=None, K = 3):
    """ Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1,3], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv4', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        assert(K==3)
        weights = tf.get_variable('weights', [128, 3*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = tf.get_variable('biases', [3*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32) + tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    #transform = tf_util.fully_connected(net, 3*K, activation_fn=None, scope='tfc3')
    transform = tf.reshape(transform, [batch_size, 3, K])
    return transform


def get_model(point_cloud, input_label, is_training, cat_num, part_num, \
		batch_size, num_point, weight_decay, bn_decay=None):
    """ ConvNet baseline, input is BxNx3 gray image """
    end_points = {}
    KNN = 12

    with tf.variable_scope('transform_net1') as sc:
        K = 3
        transform = get_transform(point_cloud, is_training, bn_decay, K = 3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)

    # KNN search
    knn_point = KNN_search(point_cloud_transformed, KNN=KNN, name_scope='KNN_search')  # 32 x 1024 x KNN x 3
    knn_point = tf.expand_dims(knn_point, axis=-1)  # 32 x 1024 x KNN x 3 x 1

    point_cloud_transformed = tf.expand_dims(point_cloud_transformed, axis=-1)
    out0 = tf_util.conv2d(point_cloud_transformed, 64, [1, 3],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training,
                          scope='conv', bn_decay=bn_decay)  # 32 x 1024 x 1 x 64
    out0_tile = tf.tile(out0, multiples=[1, 1, KNN, 1])  # 32 x 1024 x 16 x 64
    out0_tile = tf.expand_dims(out0_tile, axis=-2)

    out1 = tf_util.conv3d(knn_point, 64, [1, 1, 3],
                         padding='VALID', stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    concat = tf.concat(values=[out1, out0_tile], axis=-1)

    out2 = tf_util.conv3d(concat, 128, [1, 1, 1],
                         padding='VALID', stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    out3 = tf_util.conv3d(out2, 128, [1, 1, 1],
                         padding='VALID', stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)

    pool_k = tf_util.max_pool3d(out3, kernel_size=[1, KNN, 1],
                                stride=[1, 2, 2], padding='VALID', scope='pool_k')  # 32 x 1024 x 1 x 1 x 128
    pool_k = tf.squeeze(pool_k, axis=2)

    # VLAD layer
    vlad_out, index = VLAD(pool_k, 16, is_training, bn_decay, layer_name='VLAD')

    out4 = tf_util.conv2d(vlad_out, 512, [1, 1],
                 padding='VALID', stride=[1, 1],
                 bn=True, is_training=is_training,
                 scope='vlad_conv3', bn_decay=bn_decay)
    out5 = tf_util.conv2d(out4, 2048, [1, 1],
                 padding='VALID', stride=[1, 1],
                 bn=True, is_training=is_training,
                 scope='vlad_conv4', bn_decay=bn_decay)

    out_max = tf.nn.max_pool(out5, ksize=[1, num_point, 1, 1], strides=[1, 2, 2, 1], padding='VALID') # 32 x 1 x 1 x 1024

    # classification network
    net = tf.reshape(out_max, [batch_size, -1]) # 32 x 1 x 1024
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='cla/fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='cla/fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='cla/dp1')
    net = tf_util.fully_connected(net, cat_num, activation_fn=None, scope='cla/fc3')

    # segmentation network
    one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, 1, cat_num])
    out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])

    expand = tf.tile(out_max, [1, num_point, 1, 1])
    concat = tf.concat(axis=3, values=[expand, out0, pool_k, vlad_out, out4, out5])

    net2 = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
                        bn=True, is_training=is_training, scope='seg/conv1', weight_decay=weight_decay)
    net2 = tf_util.dropout(net2, keep_prob=0.8, is_training=is_training, scope='seg/dp1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
                        bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay)
    net2 = tf_util.dropout(net2, keep_prob=0.8, is_training=is_training, scope='seg/dp2')
    net2 = tf_util.conv2d(net2, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
                        bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay)
    net2 = tf_util.conv2d(net2, part_num, [1,1], padding='VALID', stride=[1,1], activation_fn=None, 
                        bn=False, scope='seg/conv4', weight_decay=weight_decay)

    net2 = tf.reshape(net2, [batch_size, num_point, part_num])

    return net, net2, end_points

def get_loss(l_pred, seg_pred, label, seg, weight, end_points):
    per_instance_label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l_pred, labels=label)
    label_loss = tf.reduce_mean(per_instance_label_loss)

    # size of seg_pred is batch_size x point_num x part_cat_num
    # size of seg is batch_size x point_num
    per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg), axis=1)
    seg_loss = tf.reduce_mean(per_instance_seg_loss)

    per_instance_seg_pred_res = tf.argmax(seg_pred, 2)

    total_loss = weight * seg_loss + (1 - weight) * label_loss

    return total_loss, label_loss, per_instance_label_loss, seg_loss, per_instance_seg_loss, per_instance_seg_pred_res

