[English](README.md)|[简体中文](README_cn.md)

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
- coughvid_clean_split：把筛选后的音频切分为多段，得到约2850条，并将其保存csv meta文件，写入wav音频。采样率22050，每条长度32306，也即1.465秒。
- coughvid_ml_prep：参考kaggle，预处理观察数据。
- coughvid_ml：参考kaggle，进行机器学习。
- covid19_explore：研究covid19数据集。

# VAD: Voice Activity Detection
./chapter2_VADmodel.py

# SED:Sound Event Detection
./chapter2_SEDmodel.py

# SoundDL-CoughVID
 A study repository for coughvid

# BiliCough

# NEU
