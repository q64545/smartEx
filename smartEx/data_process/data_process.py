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


file_path_debug = "/Users/ken/workrelated/smartEx/movielens_datatools/data/movielens_test"



def _parse_csv_data(raw_data, feature_size):
    record_defaults = [['0'] for _ in xrange(feature_size+1)]
    col = tf.decode_csv(records=raw_data,
                  record_defaults=record_defaults,
                  field_delim='|')
    label = tf.slice(col, begin=[0], size=[1])
    label = tf.string_to_number(label, tf.float32)
    feature = tf.slice(col, begin=[1], size=[feature_size])
    return label, feature


def inputWithDataset(file_path, batch_size=100, epochs=1, feature_size=3):
    """file_path can be the path of a file or path of a dir"""
    # 获取file_path下的文件列表
    # files = tf.train.match_filenames_once(file_path)
    files = map(lambda p: os.path.join(file_path, p), os.listdir(file_path)) if os.path.isdir(file_path) else [file_path]
    dataset = tf.data.Dataset.from_tensor_slices(files)
    # iterator0 = dataset.make_one_shot_iterator()
    dataset = dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).skip(1))
    # iterator1 = dataset.make_one_shot_iterator()
    dataset = dataset.map(lambda k: _parse_csv_data(k, feature_size))
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
    df = pd.read_csv(file_path, header=0, dtype=str, engine='c', sep='|')
    data = df.values
    data = data[:15000]
    Y_ = data[:,0:1]
    X = data[:,1:]
    return Y_, X

def inputWithPandas_batches(file_path, batch_size):
    """file_path must be the path of a file"""
    import pandas as pd
    df = pd.read_csv(file_path, header=0, dtype=str, engine='c', sep='|')
    data = df.values
    y_ = data[:, 0:1]
    x = data[:, 1:]
    X = []
    Y_ = []
    nums = x.shape[0]
    n_batches = int(nums / batch_size)

    for i in xrange(n_batches):
        X.append(x[i*batch_size: (i+1)*batch_size])
        Y_.append(y_[i*batch_size: (i+1)*batch_size])
    return Y_, X



def testContribDataset(file_path, batch_size, epochs):
    # 获取file_path下的文件列表
    # files = tf.train.match_filenames_once(file_path)
    files = map(lambda p: os.path.join(file_path, p), os.listdir(file_path))
    # dataset = tf.contrib.data.Dataset.from_tensor_slices(files)
    # dataset = dataset.flat_map(lambda filename: tf.contrib.data.TextLineDataset(filename))
    dataset = tf.contrib.data.TextLineDataset(files)
    dataset = dataset.map(_parse_csv_data)
    dataset = dataset.shuffle(buffer_size=50000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs)
    iterator_final = dataset.make_one_shot_iterator()
    next_batch_final = iterator_final.get_next()
    y_, x = next_batch_final
    return y_, x





def unit_test():
    y_, x = inputWithDataset(file_path_debug)
    with tf.Session() as sess:
        for i in range(10000):
            print(sess.run(x))



if __name__ == '__main__':
    unit_test()