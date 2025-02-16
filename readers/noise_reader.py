#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/2/16 18:00
# @Author: ZhaoKe
# @File : noise_reader.py
# @Software: PyCharm
import os
import random
import numpy as np
import librosa


def load_bilinoise_dataset(NOISE_ROOT="G:/DATAS-Medical/BILINOISE/", noise_length=22050, number=1):
    # noise_length = None
    filter_length = 25
    new_noise_list = []
    new_label_list = []
    flist = []
    for item in os.listdir(NOISE_ROOT):
        if item[-4:] == ".wav" and len(item) >= filter_length:
            flist.append(item)
    random.shuffle(flist)
    ind = 0
    for item in flist:
        cur_fname = NOISE_ROOT + item
        cur_wav, sr = librosa.load(cur_fname)
        L = len(cur_wav)
        st_pos = np.random.randint(0, L - noise_length)
        new_noise_list.append(cur_wav[st_pos:st_pos + noise_length])
        new_label_list.append(0)
        # print(NOISE_ROOT+item)
        ind += 1
        if ind == number:
            break
    return new_noise_list, new_label_list

