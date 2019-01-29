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
from abc import ABCMeta, abstractmethod
import os
import datetime
import shutil


class Train(object):
    __metaclass__ = ABCMeta

    def __init__(self, raw_param_dict):
        self.batch_size = 0
        self.use_tag = None
        self.raw_param_dict = raw_param_dict
        self.sess = None
        self.graph = tf.Graph()
        # self.graph.as_default()
        self.param_dict = None
        self.model = None
        self.sess_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            inter_op_parallelism_threads=25,
            intra_op_parallelism_threads=25,
            )
        self.sess_conf.gpu_options.allow_growth = True


    def _transform_tensorboardLogs(self, LOG_SAVE_PATH):
        name = 'events.out.tfevents'
        root = LOG_SAVE_PATH
        matches = []
        timeNow = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for cuName in os.listdir(root):
            if name in cuName:
                matches.append(os.path.join(root, cuName))
        if not os.path.exists(os.path.join(LOG_SAVE_PATH, 'backup' + timeNow)):
            os.mkdir(os.path.join(LOG_SAVE_PATH, 'backup' + timeNow))
        for m in matches:
            shutil.move(m, os.path.join(LOG_SAVE_PATH, 'backup' + timeNow))

    @abstractmethod
    def _parse_trainconf(self): pass

    @abstractmethod
    def _build_train_flow(self): pass

    @abstractmethod
    def _build_test_flow(self):
        pass

    def print_model_scale(self):
        [print(var) for var in self.graph.get_collection("trainable_variables")]
        whole_param_nums = reduce(lambda a,b: a+b, [reduce(lambda a, b: a * b, var.shape) if len(var.shape) > 0 else tf.Dimension(0) for var in self.graph.get_collection("trainable_variables")])
        print("the whole param nums is : {}".format(whole_param_nums.value))



    def run_train_flow(self):
        self._build_train_flow()

    def run_test_flow(self):
        self._build_test_flow()