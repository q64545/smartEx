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


class MLPWithEmbedding(Model):

    def __init__(self, graph, param_dict, batch_size):
        super(MLPWithEmbedding, self).__init__(graph, param_dict, batch_size)

    def build_inference(self, x, mode="train"):
        """must use LookUPSparseIDConversion"""
        initializer = self.param_dict["initializer"]
        emb_size = self.param_dict["emb_size"]
        hash_size = self.param_dict["hash_size"]
        regularizer = self.param_dict["regularizer"]
        # batch_size = self.batch_size

        with self.graph.as_default():
            with tf.variable_scope("MLP_n_Embedding"):
                with tf.variable_scope("Embedding_part"):
                    emb_v = tf.get_variable(shape=[hash_size, emb_size], regularizer=None, initializer=initializer, name="Embedding_Weights")
                    V = [tf.nn.embedding_lookup(emb_v, x_i) for x_i in x]
                    x_embedding = tf.concat(V, 1)

                with tf.variable_scope("Deep_part"):

                    hidden_layers = self.param_dict["hidden_layers"]
                    hidden_act = self.param_dict["hidden_act"]
                    fs_in = x_embedding
                    for layer in hidden_layers:
                        with tf.variable_scope("layer_nums{}_w".format(layer)):
                            w = tf.get_variable(name="weigth", shape=[fs_in.shape[1], layer], regularizer=regularizer, initializer=initializer)
                            b = tf.get_variable(name="biase", shape=[layer,], initializer=tf.constant_initializer(0.0))
                            fs_in = hidden_act(tf.matmul(fs_in, w)+b)
                    fs_out = fs_in

                with tf.variable_scope("output"):
                    output_in = fs_out
                    output_act = self.param_dict["output_act"]
                    w = tf.get_variable("output_w", shape=[output_in.shape[1], 1], regularizer=regularizer, initializer=initializer)
                    b = tf.get_variable(name="output_b", shape=[1,], initializer=tf.constant_initializer(0.0))
                    mlpemb = tf.matmul(output_in, w) + b
                    self.logit = mlpemb
                    self.prob = output_act(mlpemb)


    def get_loss(self, y_):
        with self.graph.as_default():
            with tf.name_scope("cross_entropy_loss_regularization"):
                prob = self.prob
                cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(prob, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - prob, 1e-10, 1.0)))
                regularization_loss = tf.add_n(self.graph.get_collection("regularization_losses"))
                loss = cross_entropy+regularization_loss
        return loss