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

## VAD: Voice Activity Detection
./chapter2_VADmodel.py

## SED:Sound Event Detection
./chapter2_SEDmodel.py

## ADR: Attributed based Disentangled Representation
model:
- chapter3_ADRmodel.py
- Attributed Mapper: ./modules/disentangle.py
- loss function: ./modules/loss.py
- backbone: ./models/conv_vae.py
- cls: ./models/classifiers/py

## SoundDL-CoughVID
 A study repository for coughvid

# BiliCough

# NEU
