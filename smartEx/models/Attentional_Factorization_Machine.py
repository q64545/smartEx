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


class Attentional_Factorization_Machine(Model):
    """
    This model is proposed by the paper Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks
    """

    def __init__(self, graph, param_dict, batch_size):
        super(Attentional_Factorization_Machine, self).__init__(graph, param_dict, batch_size)

    def build_inference(self, x, mode="train"):
        """must use LookUPSparseIDConversion"""
        initializer = self.param_dict["initializer"]
        k = self.param_dict["k"]
        hash_size = self.param_dict["hash_size"]
        regularizer = self.param_dict["regularizer"]


        # batch_size = self.batch_size

        with self.graph.as_default():
            with tf.variable_scope("Attentional_Factorization_Machine"):
                with tf.variable_scope("Embedding_part"):
                    emb_v = tf.get_variable(shape=[hash_size, k], regularizer=None, initializer=initializer, name="FM_Embedding_Vector")
                    V = [tf.nn.embedding_lookup(emb_v, x_i) for x_i in x]

                with tf.name_scope("Pair_Wise_Interaction_Layer"):
                    VV = []
                    for i in xrange(len(V)-1):
                        for j in xrange(i+1, len(V)):
                            VV.append(tf.multiply(V[i], V[j]))

                with tf.variable_scope("Attention_Net"):
                    at_hidden_layers = self.param_dict["at_hidden_layers"]
                    at_hidden_act = self.param_dict["at_hidden_act"]
                    in_size = k
                    AT_W = []
                    AT_B = []
                    for layer in at_hidden_layers:
                        with tf.variable_scope("AT_Net_{}".format(layer)):
                            AT_W.append(tf.get_variable(name="weight", shape=[in_size, layer], regularizer=regularizer, initializer=initializer))
                            AT_B.append(tf.get_variable(name="biase", shape=[layer,], initializer=tf.constant_initializer(0.0)))
                            in_size = layer

                    h = tf.get_variable(name="AT_h", shape=[at_hidden_layers[-1], 1], regularizer=regularizer, initializer=initializer)

                    A_score = []
                    for vv in VV:
                        at_fs_in = vv
                        for w, b in zip(AT_W, AT_B):
                            at_fs_in = at_hidden_act(tf.matmul(at_fs_in, w)+b)
                        A_score.append(tf.matmul(at_fs_in, h))

                with tf.name_scope("Attention_base_Pooling"):
                    at_pooling = tf.reduce_sum(map(lambda k: k[0] * k[1], zip(VV, A_score)), 0)

                with tf.variable_scope("AT_FM"):
                    p = tf.get_variable(name="AT_p", shape=[k, 1], regularizer=regularizer, initializer=initializer)
                    at_fm_score = tf.matmul(at_pooling, p)

                with tf.variable_scope("FM_1th"):
                    w_0 = tf.get_variable("w0", shape=[], initializer=initializer)
                    w_1 = tf.get_variable("w1", shape=[hash_size, 1], regularizer=regularizer, initializer=initializer)
                    fm1 = tf.reduce_sum(tf.nn.embedding_lookup(w_1, x), 0)
                    fm1 = w_0 + fm1

                with tf.variable_scope("output"):
                    output_act = self.param_dict["output_act"]
                    afm = at_fm_score + fm1
                    # out put shape (?, 1)
                    self.logit = afm
                    self.prob = output_act(afm)


    def get_loss(self, y_):
        with self.graph.as_default():
            with tf.name_scope("cross_entropy_loss_regularization"):
                prob = self.prob
                cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(prob, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - prob, 1e-10, 1.0)))
                regularization_loss = tf.add_n(self.graph.get_collection("regularization_losses"))
                loss = cross_entropy+regularization_loss
        return loss