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

# CoughVID Reader:
- ./readers/coughvid_reader.py
- ./readers/noise_reader.py: for data augmentation.

```text
cvr = CoughVIDReader()
sample_list, label_list = cvr.get_sample_label_list()
print("coughvid:", len(label_list), cvr.data_length)
tmplist = list(zip(sample_list, label_list))
random.shuffle(tmplist)
sample_list, label_list = zip(*tmplist)

noise_list, _ = load_bilinoise_dataset(NOISE_ROOT="./NOISE/", noise_length=bcr.data_length,
                                       number=100)
train_loader = DataLoader(
            MyDataset(audioseg=sample_list[:trte_rate], labellist=label_list[:trte_rate], noises=noise_list),
            batch_size=self.configs["batch_size"], shuffle=True)
```

# data preprocessing, ipynb:
- coughvid_clean_split: split the audio waveform into multiple segments, getting 2850 segments, then save to a csv meta file, write the sound file as wav format, the sample rating is 22050, the length of every sound is 1.465 second and the signal length is 32306.
- coughvid_ml_prep: refer to a kaggle procedure for data preprocessing.
- coughvid_ml: A machine learning method from kaggle.
- covid19_explore: research for dataset covid19.

# Data Readers:
- ./readers/*_reader.py
- ./featurizer.py: transform waveform to Mel-Spectrogram.
- ./readers/audio.py: Audio Processers.

# Machine (Deep) Learning Models for Chapter2,3,4
- chapter2_VADmodel.py: Voice Activity Detection
- chapter2_SEDmodel.py: Sound Events Detection
- chapter3_ADRmodel.py: Attributed based Disentangled Representation
- chapter4_SCDmodel.py: Sound Causality based Diagnosis
- chapter4_SCDE2Emodel.py: End-to-End SCD model

## 声音分类(VAD和SED)

MFCC特征MelSpectrogram+TDNN

框架：模型均在目录“./runs/”下。

修改train.py中的文件，调用./chapter*_*.py中的训练、测试代码。
```commandline
python chapter2_SEDmodel.py
```

### VAD: Voice Activity Detection
./chapter2_VADmodel.py

### SED:Sound Event Detection
./chapter2_SEDmodel.py

## ADR: Attributed based Disentangled Representation
model:
- chapter3_ADRmodel.py
- Attributed Mapper: ./modules/disentangle.py
- loss function: ./modules/loss.py
- backbone: ./models/conv_vae.py
- cls: ./models/classifiers/py


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
