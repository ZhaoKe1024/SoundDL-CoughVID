[English](README.md)|[简体中文](README_cn.md)
# 声音分类
This repository contains the implementation (in PyTorch) of some models for Sound/Audio Classification.

MFCC特征MelSpectrogram+TDNN
模型默认参数：数据均为9s的（频谱长度288，波形长度147000），或者10s长度309，11s长度313。采样率均为16000，自己修改参数。

框架：模型均在目录“./ackit/models/”下。

修改train.py中的文件，调用./ackit/trainer_***.py中的训练、测试代码。
```commandline
python train.py
```
暂时实现了两个模型：卷积神经网络ConvEncoder，时延神经网络TDNN。

./ackit/trainer_ConvEncoder.py，用卷积神经网络分类。输入格式(batch_size, channel, mel_length, mel_dim)

./ackit/trainer_tdnn.py，用TDNN分类。(batch_size, mel_length, mel_dim)

# Reference
1. 参考其代码结构: https://github.com/yeyupiaoling/AudioClassification-Pytorch