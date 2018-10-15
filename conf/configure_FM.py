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
    model_type = Factorization_Machine,

    # 设置训练类型
    train_type = Train_with_single_cpu,

    # 稀疏数据配置
    # feature_engineer_string_conversion = stringToOneHot,
    feature_engineer_string_conversion = LookUPSparseConversion,

    # hash_dict = [
    #              -1,        # hour
    #               8,        # C1                    7
    #               8,        # banner_pos            7
    #            5000,        # site_id               4737
    #            8000,        # site_domain           7745
    #              30,        # site_category         26
    #            9000,        # app_id                8552
    #             600,        # app_domain            559
    #              40,        # app_category          36
    #              -1,        # 2700000,        # device_id             2686408
    #              -1,        # 6800000,        # device_ip             6729486
    #            8300,        # device_model          8251
    #               6,        # device_type           5
    #               5,        # device_conn_type      4
    #            2650,        # C14                   2626
    #               9,        # C15                   8
    #              10,        # C16                   9
    #             450,        # C17                   435
    #               5,        # C18                   4
    #              70,        # C19                   68
    #             180,        # C20                   172
    #              64,        # C21                   60
    # ],

    hash_size = 2**18,

    # 对特定特征进行处理
    feature_engineer_single_feature = SingleFeature_Hour,

    single_feature_index = 0,

    # 隐向量长度k, k<<n, 一般为100以内
    k=70,

    # 参数初始化器
    initializer=tf.truncated_normal_initializer(stddev=1.0),

    # 设置优化器参数
    learning_rate=0.002,

    epochs = 2,

    max_iteration = 100000,

    # 优化算法
    optimal_algorithm=tf.train.GradientDescentOptimizer,

    # 模型和日志保存路径
    LOG_SAVE_PATH = "logs/",

    MODEL_SAVE_PATH="models_debug/",

    MODEL_NAME="model.ckpt",
)