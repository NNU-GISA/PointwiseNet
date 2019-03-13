import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from utils import tf_util
from models.transform_nets import input_transform_net
from models.VLAD_layer import VLAD, VLAD_layer
from models.KNN_search import KNN_search
from nn_sampling import sample_and_group


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    with tf.variable_scope('transform_net') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)

    # KNN local search
    #knn_point = KNN_search(point_cloud_transformed, KNN=KNN, name_scope='KNN_search')  # 32 x 1024 x KNN x 6
    #knn_point = tf.expand_dims(knn_point, axis=-1)  # 32 x 1024 x KNN x 3 x 1

    with tf.variable_scope('group_sampling') as sc:
        KNN = 16
        point_cloud_transformed, _, _, nn_points = sample_and_group(npoint=1024, radius=0.1, nsample=KNN,
                                                         xyz=point_cloud_transformed, points=None, knn=True, use_xyz=False)

    point_cloud_transformed = tf.expand_dims(point_cloud_transformed, axis=-1)
    net1 = tf_util.conv2d(point_cloud_transformed, 64, [1, 3],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1_1', bn_decay=bn_decay)   # 32 x 1024 x 1 x 64
    net1 = tf.tile(net1, multiples=[1, 1, KNN, 1])          # 32 x 1024 x 16 x 64
    net1 = tf.expand_dims(net1, axis=-2)

    nn_points = tf.expand_dims(nn_points, axis=-1) # 32 x 1024 x 16 x 3 x 1
    net = tf_util.conv3d(nn_points, 64, [1, 1, 3],
                         padding='VALID', stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1_2', bn_decay=bn_decay) # 32 x 1024 x 16 x 1 x 64
    concat = tf.concat(values=[net, net1], axis=-1)

    net = tf_util.conv3d(concat, 128, [1, 1, 1],
                         padding='VALID', stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv3d(net, 128, [1, 1, 1],
                         padding='VALID', stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    # local pooling: merge local feature
    pool_k = tf_util.max_pool3d(net, kernel_size=[1, KNN, 1],
                                stride=[1, 2, 2], padding='VALID', scope='pool_k')  # 32 x 1024 x 1 x 1 x 128
    net1 = tf.squeeze(pool_k, axis=2)

    # VLAD layer
    net2, index = VLAD(net1, 16, is_training, bn_decay, layer_name='VLAD_layer1')

    net = VLAD_layer(net2, net1, 16, is_training, bn_decay, layer_name='VLAD_layer2')

    # classification
    with tf.name_scope('fc_layer'):
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                              scope='dp1')
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                              scope='dp2')
        net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, index


def get_loss(pred, label):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
