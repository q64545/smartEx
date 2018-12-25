# —*- coding:utf-8 -*-
from __future__ import absolute_import  # 加入绝对引入特性，2.4及以前的版本默认相对引入
from __future__ import division         # 加入精确除法特性
from __future__ import print_function   # 加入该特性后使用print应该加括号
# __future__这个包旨在加入后续版本的语言特性
__author__ = "zeng pan"

from sys import version_info
if version_info.major != 2 and version_info.minor != 7:
    raise Exception('请使用Python 2.7')


from smartEx.models.Factorization_Machine import *
from smartEx.models.Fieldaware_Factorization_Machine import *
from smartEx.models.Logistic_regression import *
from smartEx.models.DeepFM import *
from smartEx.models.MLPWithEmbedding import *
from smartEx.models.Neural_Factorization_Machine import *
from smartEx.models.Wide_n_Deep import *
from smartEx.models.Attentional_Factorization_Machine import *
from smartEx.models.Deep_n_Cross import *