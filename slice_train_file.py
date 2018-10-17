# —*- coding:utf-8 -*-
# from __future__ import absolute_import  # 加入绝对引入特性，2.4及以前的版本默认相对引入
from __future__ import division         # 加入精确除法特性
from __future__ import print_function   # 加入该特性后使用print应该加括号
# __future__这个包旨在加入后续版本的语言特性
__author__ = "zeng pan"

from sys import version_info
if version_info.major != 2 and version_info.minor != 7:
    raise Exception('请使用Python 2.7')

import pandas as pd
from pandas import DataFrame




def split_train_n_test():
    file_path = "smartEx/data/train"
    df = pd.read_csv(file_path, header=0)
    data = df.values
    nums = data.shape[0]
    test_size = 50000
    x_train = data[:nums-test_size]
    x_test = data[nums-test_size:]

    df_train = DataFrame(x_train)
    df_test = DataFrame(x_test)

    # write into csv file
    df_train.to_csv("smartEx/data/kaggle_data_train", header=df.columns, index=False)
    df_test.to_csv("smartEx/data/kaggle_data_test", header=df.columns, index=False)






def main():
    small_file_size = 100000
    train_file = "smartEX/data/train"
    buffer = []
    file_index = 0
    with open(train_file, "rb") as f:
        while 1:
            line = f.readline()
            if line:
                print(line)
                buffer.append(line)
            else:
                if len(buffer) > 0:
                    with open("smartEX/data/train_small_file{}".format(file_index), "wb") as ff:
                        ff.writelines(buffer)
                    break
                else:
                    break
            if len(buffer) >= small_file_size:
                with open("smartEX/data/train_small_file{}".format(file_index), "wb") as ff:
                    ff.writelines(buffer)
                file_index += 1
                buffer = []




if __name__ == '__main__':
    split_train_n_test()