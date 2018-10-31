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


class SingleFeature_Hour(FeatureConversion):
    def __init__(self, graph, param_dict, use_tag, batch_size):
        super(SingleFeature_Hour, self).__init__(graph, param_dict, use_tag, batch_size)

    def transform(self, x):
        single_feature_index = self.param_dict["single_feature_index"]
        with self.graph.as_default():
            with tf.name_scope("feature_hour"):
                x_hour = x[:, single_feature_index]
                x_hour = tf.string_to_number(x_hour, out_type=tf.int32) % 100
                x_hour_one_hot = tf.one_hot(x_hour, depth=24)
        return x_hour_one_hot