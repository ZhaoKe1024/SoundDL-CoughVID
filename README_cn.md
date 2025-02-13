[English](README.md)|[简体中文](README_cn.md)

- ./commandcode.py：采用FFMPEG切分白噪声数据。
- ./bilicough_segment.ipynb: 
  - 采用ffmpeg处理咳嗽声音视频。
  - 根据ass数据创建csv文件。
  - 根据规则简化ass标注。
  - 根据标注统计cough数据的长度。
  - 根据csv数据截取数据及其标注（有效、无效片段二分类标注）。
  - -------=====--------
  - 根据同样规则获取切分后的噪声数据填充二分类数据集。

# VAD
通过简单的TDNN+CNN+MLP进行声音二分类：chapter2_VADmodel.py。

采用的数据集就bilicough


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

# ipynb:
数据处理和保存记录
- soundexplore: 
  - 合并status和statusSSL列，便得到./datasets/waveinfo_labedfine_staaSSL.csv
  - 提出所有标注列，得到./datasets/waveinfo_labedfine_forcls.csv
  - ./datasets/waveinfo.csv, 包含列["filename", "cough_detected", "duration", "status", "status_SSL", "anomaly"]。
  - ./datasets/waveinfo_labeled.csv: anomaly有标注
  - ./datasets/waveinfo_unlabeled.csv: anomaly无标注
  - ./datasets/waveinfo_labeled_fine.csv: anomaly有标注且duration在0.36到13s之间，且"cough_detected">0.35
  - ./datasets/waveinfo_unlabeled_fine.csv: anomaly无标注且duration在0.36到13s之间，且"cough_detected">0.35
  - F:/DATAS/COUGHVID-public_dataset_v3/waveinfo.csv: 按上一行所述条件过滤后，包含filename, cough_detected, nframes, duration, status
- coughvid_clean_split: 
  - 读取metadata_compiled.csv内容，遍历所有音频
  - 统计标注数目：COVID-19 1315, healthy 15476, symptomatic 3873
  - waveinfo_fewtoml.csv: 筛选后的720条
  - waveinfo_fewtoml_split.csv: 720条切分后的2850条。 以及写入的等长度的音频文件。
  - 读取waveinfo_fewtoml_split.csv内容，共2850行9列，uuid, status, 和专家标注7类
- coughvid_ml_prep.ipynb: 复现kaggle数据统计学描述
- coughvid_ml.ipynb: 复现kaggle机器学习攻略 
- coughvid_gm_ae: VAE训练攻略，采用数据集from readers.coughvid_reader import CoughVID_NormalAnomaly, CoughVID_Dataset
