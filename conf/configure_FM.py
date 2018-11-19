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
    data_train_path = "movielens_datatools/data/movielens_train",

    data_test_path = "movielens_datatools/data/movielens_test",
    # 定义特征个数
    feature_nums = 3,
    # 数据入口方法
    data_input_fn_train = inputWithDataset,

    data_input_fn_test = inputWithPandas_batches,
    # 设置批量大小
    batch_size = 200,

    batch_size_eval = 200,
    # 设置模型类型
    model_type = Factorization_Machine,

    # 设置训练类型
    train_type = Train_with_cpu,

    # 特征数据流设置
    feature_pipeline_type = FeatureSingeConversionPipeline,

    # 稀疏数据配置
    feature_conversion = LookUPSparseIDConversion,

    hash_size = 2**20,

    # # 对特定特征进行处理
    # feature_engineer_single_feature = SingleFeature_Hour,
    #
    # single_feature_index = 0,

    # 隐向量长度k, k<<n, 一般为100以内
    k=8,

    # 参数初始化器
    initializer=tf.truncated_normal_initializer(stddev=1.0),

    # 正则项惩罚
    regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=0.0, scale_l2=0.01),

    # 设置优化器参数
    learning_rate=0.1,

    epochs = 1000,

    max_iteration = 100000,

    # 优化算法
    optimal_algorithm=tf.train.AdamOptimizer,

    # 模型和日志保存路径
    LOG_SAVE_PATH = "logs/",

    MODEL_SAVE_PATH="models_debug/",

    MODEL_NAME="model.ckpt",
)