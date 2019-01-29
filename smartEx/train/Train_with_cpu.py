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
from smartEx.train.Train import *
from smartEx.conf_parse import *
import os
import time
import sys
import numpy as np
from sklearn import metrics


class Train_with_cpu(Train):
    """
    模型的单cpu训练流程
    """
    def __init__(self, raw_param_dict):
        super(Train_with_cpu, self).__init__(raw_param_dict)
        # 获取模型训练参数
        self.param_dict = self._parse_trainconf()

    def _parse_trainconf(self):
        return self.raw_param_dict

    def _set_data_input(self):
        """mode includes train, test and apply"""
        use_tag = self.use_tag
        if use_tag == "train":
            # 获取模型数据入口参数
            data_train_path = self.param_dict["data_train_path"]
            data_input_fn_train = self.param_dict["data_input_fn_train"]
            batch_size = self.param_dict["batch_size"]
            epochs = self.param_dict["epochs"]
            feature_nums = self.param_dict["feature_nums"]
            # 设置数据入口
            with self.graph.as_default():
                y_, x = data_input_fn_train(data_train_path, batch_size, epochs, feature_nums)
            return y_, x
        elif use_tag == "test":
            # get the param
            data_test_path = self.param_dict["data_test_path"]
            data_input_fn_test = self.param_dict["data_input_fn_test"]
            batch_size_eval = self.param_dict["batch_size_eval"]
            Y_, X = data_input_fn_test(data_test_path, batch_size_eval)
            with self.graph.as_default():
                x = tf.placeholder(dtype=tf.string, shape=[batch_size_eval, X[0].shape[1]], name='test_input_x')
                y_ = tf.placeholder(dtype=tf.float32, shape=[batch_size_eval, 1], name='test_input_y')
            return Y_, X, y_, x
        else:
            pass


    def _feature_engineer(self, x_raw):
        # 对数据进行特征工程
        # 对稀疏特征转换one hot
        # use_tag = self.use_tag
        # feature_engineer_string_to_one_hot = self.param_dict["feature_engineer_string_conversion"]
        # feature_enginerr = feature_engineer_string_to_one_hot(self.graph, self.param_dict, use_tag, self.batch_size)
        # x = feature_enginerr.transform(x_raw)

        feature_engineer_type = self.param_dict["feature_pipeline_type"]
        feature_enginerr = feature_engineer_type(self.graph, self.param_dict, self.use_tag, self.batch_size)
        x = feature_enginerr.transform(x_raw)
        return x


    def _inference(self, x, y_, mode="train"):
        # 构建前向传播计算图
        model_type = self.param_dict["model_type"]
        # 设定所示用模型
        self.model = model_type(self.graph, self.param_dict, self.batch_size)
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
        self.use_tag = "train"
        # 获取必要参数
        LOG_SAVE_PATH = self.param_dict["LOG_SAVE_PATH"]
        MODEL_SAVE_PATH = self.param_dict["MODEL_SAVE_PATH"]
        MODEL_NAME = self.param_dict["MODEL_NAME"]
        self.batch_size = self.param_dict["batch_size"]

        if not os.path.exists(LOG_SAVE_PATH):
            os.mkdir(LOG_SAVE_PATH)

        if not os.path.exists(MODEL_SAVE_PATH):
            os.mkdir(MODEL_SAVE_PATH)

        with self.graph.device("/cpu:1"):
            # 获取模型数据入口
            y_, x_raw = self._set_data_input()

            # 对输入进行特征工程
            x = self._feature_engineer(x_raw)

            # 构建前向传播计算图
            y, loss = self._inference(x, y_, "train")

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
                    # 输出模型参数量大小
                    self.print_model_scale()

                    for step in xrange(max_iteration):
                        # debug: begin
                        # debug: end
                        if step != 0 and step % 50 == 0:
                            start_time = time.time()
                            _, loss_value = self.sess.run([train_op, loss])
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
        self.use_tag = "test"
        # import pandas as pd
        # 获取必要参数
        MODEL_SAVE_PATH = self.param_dict["MODEL_SAVE_PATH"]
        self.batch_size = self.param_dict["batch_size_eval"]

        # get test data
        Y_, X, y_, x = self._set_data_input()
        YY_ = reduce(lambda a, b: np.r_[a, b], Y_).astype(int)

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
                    Y = []
                    LOSS = []
                    PCTR_MEAN = []
                    for x_i, y_i in zip(X, Y_):
                        y_v_i, loss_v, pctr_mean_v = self.sess.run([y, loss, pctr_mean], feed_dict={y_: y_i, x: x_i})
                        Y.append(y_v_i)
                        LOSS.append(loss_v)
                        PCTR_MEAN.append(pctr_mean_v)
                    Y = reduce(lambda a, b: np.r_[a, b], Y)
                    LOSS = reduce(lambda a, b: np.r_[a, b] ,LOSS)
                    fpr, tpr, thresholds = metrics.roc_curve(y_true=YY_.flatten(), y_score=Y.flatten())
                    auc_scroe = metrics.auc(fpr, tpr)


                    print("After %s training step(s), AUC score = %g, loss = %g" % (global_step, auc_scroe, LOSS.flatten().mean()))
                else:
                    print('No checkpoint file found')