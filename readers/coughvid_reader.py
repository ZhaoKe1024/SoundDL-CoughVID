# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-01-24 18:17
import os
import numpy as np
from tqdm import tqdm
import random
from torch.utils.data import Dataset
from readers.audio import AudioSegment, wav_padding
# from readers.featurizer import wav_slice_padding


def CoughVID_Class(filename="./datasets/waveinfo_labedfine_forcls.csv", isdemo=False):
    train_x, train_y = [], []
    test_x, test_y = [], []
    # label_list = []
    # train_list, test_list = [], []
    cnt = [0] * 3
    with open(filename, 'r') as fin:
        fin.readline()
        line = fin.readline().strip()
        ind = 0
        while line:
            parts = line.split(',')
            s = int(parts[7])
            cnt[s] += 1
            if cnt[s] < 100:
                # test_list.append(ind)
                test_x.append(parts[1])
                test_y.append(s)
            else:
                # train_list.append(ind)
                train_x.append(parts[1])
                train_y.append(s)
            line = fin.readline().strip()
            ind += 1
            if isdemo:
                if ind > 64:
                    break
    print("num of trainingset: ", len(train_x), len(train_y))
    print("num of testingset:", len(test_x), len(test_y))
    return train_x, train_y, test_x, test_y


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
        # tmpseg = copy(self.wav_list[ind])  # 可能造成了内存爆炸的问题？？？
        tmpseg = self.wav_list[ind]
        if len(tmpseg) > 48000:
            start_time = random.randint(0, len(tmpseg) - 48000)
            tmpseg = tmpseg[start_time:start_time+48000]
        if len(tmpseg) < 48000:
            tmpseg = wav_padding(tmpseg)
        assert len(tmpseg) == 48000, "Error Length"
        return tmpseg, self.label_list[ind]

    def __len__(self):
        return len(self.path_list)

    def append_wav(self, file_path):
        audioseg = AudioSegment.from_file(file_path)
        audioseg.vad()
        audioseg.resample(target_sample_rate=16000)
        self.wav_list.append(audioseg.samples)
        # return audioseg.samples


if __name__ == '__main__':
    # # ext_list()
    # # stat_coughvid()
    # label_list = read_labels_from_csv()
    # print(label_list.shape)
    # from torch.utils.data import DataLoader
    # from ackit.data_utils.collate_fn import collate_fn
    # from ackit.data_utils.featurizer import Wave2Mel
    plist, llist = CoughVID_Class()
    # cough_dataset = CoughVID_Dataset(plist, llist)
    # w2m = Wave2Mel(sr=16000, n_mels=80)
    # train_loader = DataLoader(cough_dataset, batch_size=32, shuffle=False,
    #                           collate_fn=collate_fn)
    # for i, (x_wav, y_label, max_len_rate) in enumerate(train_loader):
    #     print(x_wav.shape)
    #     print(y_label)
    #     print(max_len_rate)
    #     x_mel = w2m(x_wav)
    #     print(x_mel[0])
    #     break
    # print(cough_dataset.__getitem__(15084))
