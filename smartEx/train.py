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
import time
import sys

class train(object):
    __metaclass__ = ABCMeta

    def __init__(self):
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

    def run_train_flow(self):
        self._build_train_flow()

    def run_test_flow(self):
        self._build_test_flow()


class train_FM_single_cpu(train):
    """
    因子分解机模型的训练流程（train flow of factorization machine）
    """
    def __init__(self):
        super(train_FM_single_cpu, self).__init__()
        # 获取模型训练参数
        self.param_dict = self._parse_trainconf()

    def _parse_trainconf(self):
        from smartEx.trainconf import trainconf_FM
        return trainconf_FM


    def _set_data_input(self, mode="train"):
        """mode includes train, test and apply"""
        if mode == "train":
            # 获取模型数据入口参数
            data_train_path = self.param_dict["data_train_path"]
            data_input_fn_train = self.param_dict["data_input_fn_train"]
            batch_size = self.param_dict["batch_size"]
            epochs = self.param_dict["epochs"]
            # 设置数据入口
            with self.graph.as_default():
                y_, x = data_input_fn_train(data_train_path, batch_size, epochs)
            return y_, x
        elif mode == "test":
            # get the param
            data_test_path = self.param_dict["data_test_path"]
            data_input_fn_test = self.param_dict["data_input_fn_test"]
            Y_, X = data_input_fn_test(data_test_path)
            with self.graph.as_default():
                x = tf.placeholder(dtype=tf.string, shape=[None, X.shape[1]], name='test_input_x')
                y_ = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='test_input_y')
            return Y_, X, y_, x
        else:
            pass


    def _feature_engineer(self, x):
        # 对数据进行特征工程
        # 对稀疏特征转换one hot
        feature_engineer_string_to_one_hot = self.param_dict["feature_engineer_string_to_one_hot"]
        feature_enginerr = feature_engineer_string_to_one_hot(self.graph, self.param_dict)
        x_one_hot = feature_enginerr.transform(x)
        # 对特征hour进行特殊处理
        feature_engineer_single_feature = self.param_dict["feature_engineer_single_feature"]
        feature_enginerr2 = feature_engineer_single_feature(self.graph, self.param_dict)
        x_hour = feature_enginerr2.transform(x)
        return tf.concat([x_hour, x_one_hot], 1)


    def _inference(self, x, y_, mode="train"):
        # 构建前向传播计算图
        model_type = self.param_dict["model_type"]
        # 设定所示用模型
        self.model = model_type(self.graph, self.param_dict)
        self.model.build_inference(x, mode)
        loss = self.model.get_loss(y_)
        return self.model.prob, loss


    def _get_train_op(self, loss):
        # 获取模型训练参数
        learning_rate = self.param_dict["learning_rate"]
        optimal_algorithm = self.param_dict["optimal_algorithm"]
        with self.graph.as_default():
            train_op = optimal_algorithm(learning_rate).minimize(loss)
        return train_op


    def _build_train_flow(self):
        # 获取必要参数
        LOG_SAVE_PATH = self.param_dict["LOG_SAVE_PATH"]
        MODEL_SAVE_PATH = self.param_dict["MODEL_SAVE_PATH"]
        MODEL_NAME = self.param_dict["MODEL_NAME"]

        if not os.path.exists(LOG_SAVE_PATH):
            os.mkdir(LOG_SAVE_PATH)

        if not os.path.exists(MODEL_SAVE_PATH):
            os.mkdir(MODEL_SAVE_PATH)

        with self.graph.device("/cpu:1"):
        # 获取模型数据入口
            y_, x = self._set_data_input("train")

            # 对输入进行特征工程
            x_one_hot = self._feature_engineer(x)

            # 构建前向传播计算图
            y, loss = self._inference(x_one_hot, y_, "train")

            # 获取训练OP
            train_op = self._get_train_op(loss)

            # 添加监控
            with self.graph.as_default():
                tf.summary.scalar('total_loss', loss)

            # 创建剩余OP
            saver = tf.train.Saver(self.graph.get_collection("variables"))

            # init = self.graph
            init_ops = map(lambda v: v.initializer, self.graph.get_collection("variables"))

            local_init_ops =map(lambda v: v.initializer, self.graph.get_collection("local_variables"))

            with self.graph.as_default():
                summary_op = tf.summary.merge_all()

            # 创建会话，开始训练流程
            max_iteration = self.param_dict["max_iteration"]
            self.sess = tf.Session(config=self.sess_conf, graph=self.graph)
            try:
                with self.sess.as_default():
                    self.sess.run(local_init_ops)
                    self.sess.run(init_ops)
                    self._transform_tensorboardLogs(LOG_SAVE_PATH)
                    summary_writer = tf.summary.FileWriter(LOG_SAVE_PATH, self.sess.graph)

                    for step in xrange(max_iteration):
                        # debug: begin
                        # debug: end
                        if step != 0 and step % 50 == 0:
                            start_time = time.time()
                            _, loss_value = self.sess.run([train_op,loss])
                            duration = time.time() - start_time
                            print("step {}, loss = {} ({} sec/batch)".format(step, loss_value, duration))
                            summary = self.sess.run(summary_op)
                            summary_writer.add_summary(summary, step)
                        else:
                            _ = self.sess.run(train_op)
                        if step % 50 == 0 or (step + 1) == max_iteration:
                            checkpoint_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
                            saver.save(self.sess, checkpoint_path, global_step=step)
            except tf.errors.OutOfRangeError:
                print("train ended!")
            except Exception as e:
                print("Exception type:%s" % type(e))
                print("Unexpected Error: {}".format(e))
                sys.exit(1)

            self.sess.close()


    def _build_test_flow(self):
        import pandas as pd
        # 获取必要参数
        MODEL_SAVE_PATH = self.param_dict["MODEL_SAVE_PATH"]

        # get test data
        Y_, X, y_, x = self._set_data_input("test")

        # 对输入进行特征工程
        x_one_hot = self._feature_engineer(x)

        # 构建前向传播计算图
        y, loss = self._inference(x_one_hot, y_, "test")

        # 计算评价指标
        with self.graph.as_default():
            auc, update_op = tf.metrics.auc(y_, y)
            pctr_mean = tf.reduce_mean(y, 0)

        # 读取已训练模型参数
        saver = tf.train.Saver(self.graph.get_collection("variables"))
        local_init_ops = map(lambda v: v.initializer, self.graph.get_collection("local_variables"))

        # 创建会话，开始测试流程
        self.sess = tf.Session(config=self.sess_conf, graph=self.graph)

        with self.sess.as_default():
            self.sess.run(local_init_ops)
            while 1:
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    auc_v, loss_v, pctr_mean_v = self.sess.run([auc, loss, pctr_mean], feed_dict={y_: Y_, x: X})
                    print("After %s training step(s), AUC score = %g, loss = %g, mean pctr = %g" % (global_step, auc_v, loss_v, pctr_mean_v))
                else:
                    print('No checkpoint file found')





















