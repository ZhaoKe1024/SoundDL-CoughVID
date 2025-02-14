#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/12/4 15:54
# @Author: ZhaoKe
# @File : test_code.py
# @Software: PyCharm
"""
暂时测试代码用
"""

import torch
import torchaudio


def redundancy_overlap_generate():
    length = 46
    data_length = 11
    cnt_sum = length // data_length + 1
    res = cnt_sum * data_length - length
    print(cnt_sum, res)
    overlap = res // (cnt_sum - 1)
    print(overlap)
    st = 0
    while st + data_length <= length:
        print("[{}, {}]".format(st, st + data_length))
        st += data_length - overlap
    print("[{}, {}]".format(length - data_length - 1, length - 1))


def min2sec(t: str):
    parts = t.split(':')
    res = float(parts[-1])
    f = 60
    for i in range(len(parts) - 1):
        res += int(parts[len(parts) - 2 - i]) * f
        f *= 60
    return res


class Wave2Mel(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        return self.amplitude_to_db(self.mel_transform(x))


def find_wavlength():
    w2m = Wave2Mel(sr=22050)
    for length in range(22050, 33075, 1000):
        print(w2m(torch.rand(4, length)).shape)


if __name__ == '__main__':
    # redundancy_overlap_generate()
    # print(min2sec("00:01:19.06")*22050)
    # find_wavlength()
    pass
