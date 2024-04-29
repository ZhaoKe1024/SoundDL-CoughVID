#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/28 16:06
# @Author: ZhaoKe
# @File : deelenc.py
# @Software: PyCharm
import torch
from pretrained.wav2vec import Wav2Vec


if __name__ == '__main__':
    encoder = Wav2Vec(pretrained=True)
    print(encoder)
    # x = torch.randn(16, 16000)  # [1, 16000]
    # z = encoder(x)  # [1, 512, 98]
    # print(z.shape)
    # x = torch.randn(1, 32000)  # [1, 32000]
    # z = encoder(x)  # [1, 512, 198]
    # print(z.shape)
    x = torch.randn(16, 48000)  # [16, 48000]
    z = encoder(x)  # [16, 512, 298]
    print(z.shape)
