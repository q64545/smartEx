# —*- coding:utf-8 -*-
from __future__ import absolute_import  # 加入绝对引入特性，2.4及以前的版本默认相对引入
from __future__ import division         # 加入精确除法特性
from __future__ import print_function   # 加入该特性后使用print应该加括号
# __future__这个包旨在加入后续版本的语言特性
__author__ = "zeng pan"

from smartEx.train import *
import os

def main():
    train_flow = train_FM_single_cpu()
    train_flow.run_test_flow()


if __name__ == "__main__":
    print(os.getcwd())
    main()