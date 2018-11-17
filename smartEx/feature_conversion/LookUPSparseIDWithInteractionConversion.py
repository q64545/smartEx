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


class LookUPSparseIDWithInteractionConversion(FeatureConversion):
    def __init__(self, graph, param_dict, use_tag, batch_size):
        super(LookUPSparseIDWithInteractionConversion, self).__init__(graph, param_dict, use_tag, batch_size)

    def transform(self, x):
        """
        :param x: tensor of string, E.g "a,b,c"
        :return:  dict of ids(tensor) lists, E.g {sparse=[tensor(123), tensor(342), tensor(532)], inter_sparse=[tensor(12), tensor(32), tensor(52)]}
        """
        hash_size = self.param_dict["hash_size"]
        feature_interaction = self.param_dict["feature_interaction"]
        with self.graph.as_default():
            x_sparse = []
            x_interact_sparse = []
            with tf.name_scope("sparse_lookup"):
                for i in xrange(x.shape[1].value):
                    x_i = tf.string_to_hash_bucket_fast(input="f{}_".format(i) + x[:, i], num_buckets=hash_size, name="sparse_feature_{}".format(i))
                    x_sparse.append(x_i)
            with tf.name_scope("inter_sparse_lookup"):
                for ixj in feature_interaction:
                    i, j = ixj.split('x')
                    x_ixj = x[:,int(i)] +"x" +  x[:, int(j)]
                    x_i = tf.string_to_hash_bucket_fast(input="f{}x{}_".format(i, j)+x_ixj, num_buckets=hash_size, name="sparse_feature_interact_{}x{}".format(i, j))
                    x_interact_sparse.append(x_i)
        return {"sparse" : x_sparse, "inter_sparse" : x_interact_sparse}
