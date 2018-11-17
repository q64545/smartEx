# —*- coding:utf-8 -*-
from __future__ import absolute_import  # 加入绝对引入特性，2.4及以前的版本默认相对引入
from __future__ import division         # 加入精确除法特性
from __future__ import print_function   # 加入该特性后使用print应该加括号
# __future__这个包旨在加入后续版本的语言特性
__author__ = "zeng pan"

from sys import version_info
if version_info.major != 2 and version_info.minor != 7:
    raise Exception('请使用Python 2.7')

import tensorflow as tf
from smartEx.feature_conversion.FeatureConversion import *


class LookUPSparseConversion(FeatureConversion):
    def __init__(self, graph, param_dict, use_tag, batch_size):
        super(LookUPSparseConversion, self).__init__(graph, param_dict, use_tag, batch_size)

    def transform(self, x):
        hash_size = self.param_dict["hash_size"]
        batch_size = self.batch_size
        with self.graph.as_default():
            x_sparse = []
            with tf.name_scope("sparse_lookup"):
                for i in xrange(x.shape[1].value):
                    x_i = tf.string_to_hash_bucket_fast(input="f{}_".format(i) + x[:, i], num_buckets=hash_size, name="sparse_feature_{}".format(i))
                    x_sparse.append(x_i)

            dense_size = batch_size * x.shape[1].value
            indice = []
            for i in xrange(x.shape[1].value):
                indice_i = tf.concat([tf.constant(range(batch_size), shape=[batch_size, 1], dtype=tf.int64), tf.reshape(x_sparse[i], shape=[batch_size, 1])], 1)
                indice.append(indice_i)

            indice_tensor = tf.concat(indice, 0)
            x_sparse_lookup = tf.sparse_to_dense(sparse_indices=indice_tensor, sparse_values=[1.0]*dense_size, output_shape=[batch_size, hash_size], default_value=0, validate_indices=False)
        return x_sparse_lookup
