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
from smartEx.models.Model import *

class Factorization_Machine(Model):

    def __init__(self, graph, param_dict, batch_size):
        super(Factorization_Machine, self).__init__(graph, param_dict, batch_size)


    def build_inference(self, x, mode="train"):
        with self.graph.as_default():
            # 定义待学习参数 w0, w1,...,wn, v1,...,vn
            initializer = self.param_dict["initializer"]
            k = self.param_dict["k"]
            with tf.variable_scope("Factorization_Machine_inference"):
                w0 = tf.get_variable("w0", shape=[], initializer=initializer)
                w = self.get_weight_variable(shape=[x.shape[1].value, 1], regularizer=None, initializer=initializer, name="w")
                v = self.get_weight_variable(shape=[x.shape[1].value, k], regularizer=None, initializer=initializer, name="v")
                y_part1 = w0 + tf.matmul(x, w)
                sum_k = []
                for i in xrange(k):
                    v_i = v[:, i:i+1]
                    sum_k.append(tf.square(tf.matmul(x, v_i)) - tf.matmul(tf.square(x), tf.square(v_i)))
                y_part2 = 1/2 * tf.reduce_sum(sum_k, 0)
                self.logit = y_part1 + y_part2
                self.prob = tf.nn.sigmoid(self.logit)

    def get_loss(self, y_):
        with self.graph.as_default():
            with tf.name_scope("cross_entropy_loss"):
                prob = self.prob
                cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(prob, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - prob, 1e-10, 1.0)))
        return cross_entropy
