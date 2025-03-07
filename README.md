[English](README.md)|[简体中文](README_cn.md)

# data preprocessing in Jupyter:
- coughvid_clean_split: split the audio waveform into multiple segments, getting 2850 segments, then save to a csv meta file, write the sound file as wav format, the sample rating is 22050, the length of every sound is 1.465 second and the signal length is 32306.
- coughvid_ml_prep: refer to a kaggle procedure for data preprocessing.
- coughvid_ml: A machine learning method from kaggle.
- neudata_1explore.ipynb
- neudata_2silence_segment.ipynb
- neudata_3clean_split.ipynb
- bilicough_1explore.ipynb
- bilicough_2vad.ipynb
- bilicough_3sed.ipynb
- covid19_explore.ipynb: research for dataset covid19.

# Data Readers:
### dataset for c2VAD:
 about bilicough_2vad.ipynb
### dataset for c2SED:
- bilicough_metainfo_c2sed.csv: generate from bilicough_1explore.ipynb
### dataset for c4: 
- bilicough_metainfo_c4scd_250307_dura.csv: 
  - generate from the ./readers/bilicough_reader.py. 
  - def generate_SCD_metainfo(task="scd"). 
- ./datasets/metainfo4scd.json: 
  - generate from the ./readers/bilicough_reader.py
  - class BiliCoughReader(object).generate_SCD_metainfo(task="scd").
### related code:
- ./readers/*_reader.py
- ./featurizer.py: transform waveform to Mel-Spectrogram.
- ./readers/audio.py: Audio Processers.
- ./readers/coughvid_reader.py
- ./readers/noise_reader.py: for data augmentation.

### Take COUGHVID dataset as a example:
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

# Machine (Deep) Learning Models for Chapter2,3,4(Private, omitted)
- chapter2_VADmodel.py: Voice Activity Detection
- chapter2_SEDmodel.py: Sound Events Detection
- chapter3_ADRmodel.py: Attributed based Disentangled Representation
- chapter4_SCDmodel.py: Sound Causality based Diagnosis
- chapter4_SCDE2Emodel.py: End-to-End SCD model

## 1 VAD: Voice Activity Detection
./chapter2_VADmodel.py

## 2 SED:Sound Event Detection
./chapter2_SEDmodel.py

## 3 ADR: Attributed based Disentangled Representation
model:
- chapter3_ADRmodel.py
- Attributed Mapper: ./modules/disentangle.py
- loss function: ./modules/loss.py
- backbone: ./models/conv_vae.py
- cls: ./models/classifiers/py

## 4 SCD Diagnosis using GNN/Attention
./chapter4_SCDmodel.py
