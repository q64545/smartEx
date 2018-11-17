# —*- coding:utf-8 -*-
__author__ = "zeng pan"

import os
from importlib import import_module
import argparse
import tensorflow as tf

trainScript = "configure_WDL"
trainconf = import_module("conf."+trainScript).trainconf
train_flow = trainconf["train_type"](trainconf)
self = train_flow
