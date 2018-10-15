# -*- coding:utf-8 -*-
from __future__ import absolute_import  # 加入绝对引入特性，2.4及以前的版本默认相对引入
from __future__ import division         # 加入精确除法特性
from __future__ import print_function   # 加入该特性后使用print应该加括号
# __future__这个包旨在加入后续版本的语言特性
__author__ = "zeng pan"

from sys import version_info
if version_info.major != 2 and version_info.minor != 7:
    raise Exception('请使用Python 2.7')


import tensorflow as tf
import numpy as np
import os


# file_path_debug = ["/Users/zengpan1/ads_sz_dev.data_proj/datahouse/python/biz/local_ctr_project/kaggle_datasets/data/train"]



def _parse_csv_data(raw_data):
    record_defaults = [['0'] for _ in xrange(24)]
    col = tf.decode_csv(records=raw_data,
                  record_defaults=record_defaults,
                  field_delim=',')
    label = tf.slice(col, begin=[1], size=[1])
    label = tf.string_to_number(label, tf.float32)
    feature = tf.slice(col, begin=[2], size=[22])
    return label, feature


def inputWithDataset(file_path, batch_size, epochs):
    """file_path can be the path of a file or path of a dir"""
    # 获取file_path下的文件列表
    # files = tf.train.match_filenames_once(file_path)
    files = map(lambda p: os.path.join(file_path, p), os.listdir(file_path)) if os.path.isdir(file_path) else [file_path]
    dataset = tf.data.Dataset.from_tensor_slices(files)
    # iterator0 = dataset.make_one_shot_iterator()
    dataset = dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).skip(1))
    # iterator1 = dataset.make_one_shot_iterator()
    dataset = dataset.map(_parse_csv_data)
    # iterator2 = dataset.make_one_shot_iterator()
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    # iterator3 = dataset.make_one_shot_iterator()
    dataset = dataset.repeat(epochs)

    iterator_final = dataset.make_one_shot_iterator()
    next_batch_final = iterator_final.get_next()
    y_, x = next_batch_final
    return y_, x

def inputWithPandas(file_path):
    """file_path must be the path of a file"""
    import pandas as pd
    df = pd.read_csv(file_path, header=0, dtype=str, engine='c')
    data = df.values
    data = data[:15000]
    Y_ = data[:,1:2]
    X = data[:,2:]
    return Y_, X

def inputWithPandas_batches(file_path, batch_size):
    """file_path must be the path of a file"""
    import pandas as pd
    df = pd.read_csv(file_path, header=0, dtype=str, engine='c')
    data = df.values
    data = data[:15000]
    y_ = data[:, 1:2]
    x = data[:, 2:]
    X = []
    Y_ = []
    # waiting for developing ........................
    # waiting for developing ........................
    # waiting for developing ........................
    # waiting for developing ........................
    return Y_, X



# def testContribDataset(file_path, batch_size, epochs):
#     # 获取file_path下的文件列表
#     # files = tf.train.match_filenames_once(file_path)
#     files = map(lambda p: os.path.join(file_path, p), os.listdir(file_path))
#     # dataset = tf.contrib.data.Dataset.from_tensor_slices(files)
#     # dataset = dataset.flat_map(lambda filename: tf.contrib.data.TextLineDataset(filename))
#     dataset = tf.contrib.data.TextLineDataset(files)
#     dataset = dataset.map(_parse_csv_data)
#     dataset = dataset.shuffle(buffer_size=50000)
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.repeat(epochs)
#     iterator_final = dataset.make_one_shot_iterator()
#     next_batch_final = iterator_final.get_next()
#     y_, x = next_batch_final
#     return y_, x





def unit_test():
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0,2.0,3.0,4.0,5.0]))

    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(one_element))



if __name__ == '__main__':
    unit_test()