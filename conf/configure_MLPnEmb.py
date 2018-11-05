# —*- coding:utf-8 -*-
from __future__ import absolute_import  # 加入绝对引入特性，2.4及以前的版本默认相对引入
from __future__ import division         # 加入精确除法特性
from __future__ import print_function   # 加入该特性后使用print应该加括号
# __future__这个包旨在加入后续版本的语言特性
__author__ = "zeng pan"

from sys import version_info
if version_info.major != 2 and version_info.minor != 7:
    raise Exception('请使用Python 2.7')

from smartEx import *

trainconf = dict(
    # 数据入口
    data_train_path = "data/kaggle_data_train",

    data_test_path = "data/kaggle_data_test",
    # 数据入口方法
    data_input_fn_train = inputWithDataset,

    data_input_fn_test = inputWithPandas_batches,
    # 设置批量大小
    batch_size = 200,

    batch_size_eval = 200,
    # 设置模型类型
    model_type = MLPWithEmbedding,

    # 设置训练类型
    train_type = Train_with_cpu,

    # 特征数据流设置
    feature_pipeline_type=FeatureSingeConversionPipeline,
    # 稀疏数据配置
    feature_conversion=LookUPSparseIDConversion,

    # 稀疏数据的最维数
    hash_size = 2**15,

    # 设置Embedding大小
    emb_size = 4,

    # deep部分配置
    hidden_layers = [258, 128, 50],
    hidden_act = tf.nn.relu,

    # 输出层配置
    output_act = tf.nn.sigmoid,

    # 正则项惩罚
    regularizer = tf.contrib.layers.l2_regularizer(0.0001),

    # 参数初始化器
    initializer=tf.truncated_normal_initializer(stddev=1.0),

    # 设置优化器参数
    learning_rate=0.1,

    # 数据的轮数
    epochs = 2,

    max_iteration = 100000,

    # 优化算法
    optimal_algorithm=tf.train.GradientDescentOptimizer,

    # 模型和日志保存路径
    LOG_SAVE_PATH = "logs/",

    MODEL_SAVE_PATH="models_debug/",

    MODEL_NAME="model.ckpt",
)