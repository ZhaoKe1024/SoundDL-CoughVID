#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/11/14 22:02
# @Author: ZhaoKe
# @File : neucough_reader.py
# @Software: PyCharm
# import sys
# sys.path.append(r'C:/Program Files (zk)/PythonFiles/AClassification/SoundDL-CoughVID')
import pandas as pd
from readers.featurizer import Wave2Mel
ROOT = "F:/DATAS/NEUCOUGHDATA_COUGHSINGLE/"


def read_metafile():
    slice_raw = pd.read_csv(ROOT + 'neucough_metainfo_slice.txt', header=0, index_col=0)
    slice_raw
