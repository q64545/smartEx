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


class Fieldaware_Factorization_Machine(Model):

    def __init__(self, graph, param_dict, batch_size):
        super(Fieldaware_Factorization_Machine, self).__init__(graph, param_dict, batch_size)

    def build_inference(self, x, mode="train"):
        # must use LookUPSparseIDConversion
        initializer = self.param_dict["initializer"]
        k = self.param_dict["k"]
        hash_size = self.param_dict["hash_size"]
        regularizer = self.param_dict["regularizer"]
        # batch_size = self.batch_size

        with self.graph.as_default():
            with tf.variable_scope("Fieldaware_Factorization_Machine_inference"):
                # define v
                num_feilds = len(x)
                v = []
                for i in xrange(num_feilds):
                    v_i = tf.get_variable(shape=[hash_size, k], regularizer=regularizer, initializer=initializer, name="filed{}_v".format(i))
                    v.append(v_i)
                V = []
                for i in xrange(num_feilds):
                    V_i = [tf.nn.embedding_lookup(v[i], x_i) for x_i in x]
                    V.append(V_i)
                VV = []
                for i in xrange(len(V)):
                    for j in xrange(i+1, len(V)):
                        VV.append(tf.reduce_sum(tf.multiply(V[j][i], V[i][j]), 1))
                ffm2 = reduce(lambda a, b: a+b, VV)
                # define w0
                w0 = tf.get_variable("w0", shape=[], initializer=initializer)
                # define w1
                w1 = tf.get_variable("w1", shape=[hash_size, 1], regularizer=regularizer, initializer=initializer)
                ffm1 = tf.reduce_sum(tf.reshape(tf.nn.embedding_lookup(w1, x), [-1, num_feilds]), 1)
                ffm = w0 + ffm1 + ffm2
                self.logit = ffm
                self.prob = tf.nn.sigmoid(ffm)


    def get_loss(self, y_):
        with self.graph.as_default():
            with tf.name_scope("cross_entropy_loss"):
                prob = self.prob
                cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(prob, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - prob, 1e-10, 1.0)))
        return cross_entropy