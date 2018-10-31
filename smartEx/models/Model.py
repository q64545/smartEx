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


class Model(object):

    __metaclass__ = ABCMeta

    def __init__(self, graph, param_dict, bstch_size):
        self.graph = graph
        self.param_dict = param_dict
        self.logit = None
        self.prob = None
        self.batch_size = bstch_size

    def get_weight_variable(self, shape, regularizer, initializer, name="weights"):
        weights = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
        if regularizer != None:
            tf.add_to_collection("regularization_losses", regularizer(weights))
        return weights

    @abstractmethod
    def build_inference(self, x): pass

    @abstractmethod
    def get_loss(self, y_): pass