import tensorflow as tf
import numpy as np


def Top_K(one_hot, top_k, batch_size, num_point):
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
    one_hot = tf.cast(one_hot[1], dtype=tf.int32)
    one_hot = tf.reshape(one_hot, shape=[batch_size, num_point, top_k, -1])
    fuse = tf.concat([concat, one_hot], axis=3)

    return fuse



def KNN_search(points, KNN, name_scope=None):
    with tf.variable_scope(name_scope):
        batch_size = points.get_shape()[0].value
        num_point = points.get_shape()[1].value

        point_cp = tf.expand_dims(points, axis=2)  # 32 x 1024 x 1 x 3
        tile = tf.tile(point_cp, [1, 1, num_point, 1])  # 32 x 1024 x 1024 x 3

        diff = tile - tf.expand_dims(points, axis=1)
        distance = tf.sqrt(tf.reduce_sum(diff ** 2, axis=-1))
        distance = tf.exp(-distance)

        diag = tf.cast(tf.diag(np.ones(num_point)), tf.float32)
        distance = distance - diag

        index = tf.expand_dims(distance, axis=-2)
        one_hot = tf.nn.top_k(index, KNN)

        fuse = Top_K(one_hot, KNN, batch_size, num_point)

        data = tf.tile(tf.expand_dims(points, axis=1), [1, num_point, 1, 1])
        knn_point = tf.gather_nd(data, fuse)
       
        difference = knn_point - point_cp
        #point_cp = tf.tile(point_cp, [1,1,KNN,1])
        #concat = tf.concat([point_cp, difference], axis=-1)

    return difference
