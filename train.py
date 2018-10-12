# —*- coding:utf-8 -*-
from __future__ import absolute_import  # 加入绝对引入特性，2.4及以前的版本默认相对引入
from __future__ import division         # 加入精确除法特性
from __future__ import print_function   # 加入该特性后使用print应该加括号
# __future__这个包旨在加入后续版本的语言特性
__author__ = "zeng pan"

# from conf.configure import trainconf
import os
from importlib import import_module
import argparse


def main(args):
    trainScript = args.trainConfure
    print("training configure is conf/{}.py".format(trainScript))
    trainconf = import_module("conf."+trainScript).trainconf    # get train configure dict

    train_flow = trainconf["train_type"](trainconf)
    train_flow.run_train_flow()


if __name__ == "__main__":
    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--trainConfure', type=str, default="configure_FM")   # get training configure
    args = parser.parse_args()
    main(args)