# —*- coding:utf-8 -*-
from __future__ import absolute_import  # 加入绝对引入特性，2.4及以前的版本默认相对引入
from __future__ import division         # 加入精确除法特性
from __future__ import print_function   # 加入该特性后使用print应该加括号
# __future__这个包旨在加入后续版本的语言特性
__author__ = "zeng pan"

from sys import version_info
if version_info.major != 2 and version_info.minor != 7:
    raise Exception('请使用Python 2.7')

from abc import ABCMeta, abstractmethod
import tensorflow as tf


class featureEngineer(object):
    __metaclass__ = ABCMeta

    def __init__(self, graph, param_dict):
        self.graph = graph
        self.param_dict = param_dict
        pass

    @abstractmethod
    def transform(self, x): pass


class stringToOneHot(featureEngineer):

    def __init__(self, graph, param_dict):
        super(stringToOneHot, self).__init__(graph, param_dict)


    def transform(self, x):
        hash_size = self.param_dict['hash_size']
        with self.graph.as_default():
            x_sparse = []
            x_one_hot = []
            with tf.name_scope("spars_to_one_hot"):
                for i in xrange(len(hash_size)):
                    if hash_size[i] != -1:
                        x_i = tf.string_to_hash_bucket_fast(input=x[:, i], num_buckets=hash_size[i],
                                                        name="sparse_feature_{}".format(i))
                        x_one_hot_i = tf.one_hot(x_i, depth=hash_size[i])
                        x_sparse.append(x_i)
                        x_one_hot.append(x_one_hot_i)
            x_one_hot = tf.concat(x_one_hot, 1)
        return x_one_hot


class singleFeature_Hour(featureEngineer):
    def __init__(self, graph, param_dict):
        super(singleFeature_Hour, self).__init__(graph, param_dict)

    def transform(self, x):
        single_feature_index = self.param_dict["single_feature_index"]
        with self.graph.as_default():
            with tf.name_scope("feature_hour"):
                x_hour = x[:, single_feature_index]
                x_hour = tf.string_to_number(x_hour, out_type=tf.int32) % 100
                x_hour_one_hot = tf.one_hot(x_hour, depth=24)
        return x_hour_one_hot




