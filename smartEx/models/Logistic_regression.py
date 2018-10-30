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


class Logistic_Regression(Model):

    def __init__(self, graph, param_dict, batch_size):
        super(Logistic_Regression, self).__init__(graph, param_dict, batch_size)

    def build_inference(self, x, mode="train"):
        # must use LookUPSparseConversion
        initializer = self.param_dict["initializer"]
        k = self.param_dict["k"]
        hash_size = self.param_dict["hash_size"]
        regularizer = self.param_dict["regularizer"]
        batch_size = self.batch_size



    def get_loss(self, y_):
        pass