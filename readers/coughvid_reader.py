# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-01-24 18:17
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import librosa
import time
from torch.utils.data import Dataset
from readers.audio import AudioSegment
from readers.featurizer import wav_slice_padding


def CoughVID_NormalAnomaly(filename="./datasets/waveinfo_labeled_fine.csv", istrain=True, isdemo=False):
    healthy_p_list = []
    unhealthy_p_list = []
    healthy_label_list = []
    unhealthy_label_list = []
    # healthy_anomaly_list = []
    # unhealthy_anomaly_list = []
    with open(filename, 'r') as fin:
        fin.readline()
        line = fin.readline().strip()
        while line:
            parts = line.split(',')
            status = int(float(parts[6]))
            if status == 0:
                healthy_p_list.append(parts[1])
                healthy_label_list.append(np.array(float(parts[6]), dtype=np.int64))
                # healthy_anomaly_list.append(np.array(0, dtype=np.int64))
            else:

                unhealthy_p_list.append(parts[1])
                unhealthy_label_list.append(np.array(float(parts[6]), dtype=np.int64))
                # unhealthy_anomaly_list.append(np.array(1, dtype=np.int64))
            line = fin.readline().strip()
    return healthy_p_list, healthy_label_list, unhealthy_p_list, unhealthy_label_list


def CoughVID_Lists(filename="../../datasets/waveinfo_annotation.csv", istrain=True, isdemo=False):
    path_list = []
    label_list = []
    with open(filename, 'r') as fin:
        fin.readline()
        line = fin.readline()
        ind = 0
        while line:
            parts = line.split(',')
            path_list.append(parts[1])
            label_list.append(np.array(parts[2], dtype=np.int64))
            line = fin.readline()
            ind += 1
            if isdemo:
                if ind > 1000:
                    return path_list, label_list
    N = len(path_list)
    tr, va = int(N * 0.8), int(N * 0.9)
    train_path, train_label = path_list[0:tr], label_list[0:tr]
    valid_path, valid_label = path_list[tr:va], label_list[tr:va]
    if istrain:
        return train_path, train_label, valid_path, valid_label
    else:
        return path_list[va:], label_list[va:]


class CoughVID_Dataset(Dataset):
    def __init__(self, path_list, label_list):
        self.path_list = path_list
        self.label_list = label_list
        self.wav_list = []
        for item in tqdm(path_list, desc="Loading"):
            self.append_wav(item)

    def __getitem__(self, ind):
        return self.wav_list[ind], self.label_list[ind]

    def __len__(self):
        return len(self.path_list)

    def append_wav(self, file_path):
        audioseg = AudioSegment.from_file(file_path)
        audioseg.vad()
        audioseg.resample(target_sample_rate=16000)
        audioseg.crop(duration=3.0, mode="train")
        audioseg.wav_padding()
        assert len(audioseg) == 48000, "Error Length"
        self.wav_list.append(audioseg.samples)


if __name__ == '__main__':
    # # ext_list()
    # # stat_coughvid()
    # label_list = read_labels_from_csv()
    # print(label_list.shape)
    from torch.utils.data import DataLoader
    from ackit.data_utils.collate_fn import collate_fn
    from ackit.data_utils.featurizer import Wave2Mel

    cough_dataset = CoughVID_Dataset()
    w2m = Wave2Mel(sr=16000, n_mels=80)
    train_loader = DataLoader(cough_dataset, batch_size=32, shuffle=False,
                              collate_fn=collate_fn)
    for i, (x_wav, y_label, max_len_rate) in enumerate(train_loader):
        print(x_wav.shape)
        print(y_label)
        print(max_len_rate)
        x_mel = w2m(x_wav)
        print(x_mel[0])
        break
    # print(cough_dataset.__getitem__(15084))
