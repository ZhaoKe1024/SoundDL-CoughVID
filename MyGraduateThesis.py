# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2025-02-06 23:37
import os
# import json
# import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
import librosa
# DATA_ROOT = "F:/DATAS/NEUCOUGHDATA_COUGH/"
DATA_ROOT = "E:/DATAS-Medical/BILIBILICOUGH/"
fname = "bilicough_004"


def preprocessing_show():
    wav_test_path = os.path.join(DATA_ROOT, fname + ".wav")
    ass_test_path = os.path.join(DATA_ROOT, fname + ".ass")
    y, sr = librosa.load(wav_test_path)
    plt.figure(0)
    plt.plot(range(len(y)), y, c="#000000")
    plt.show()


if __name__ == '__main__':
    preprocessing_show()
