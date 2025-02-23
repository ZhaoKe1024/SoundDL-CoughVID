# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-01-24 18:17
import os
import random

import numpy as np
import librosa
from torch.utils.data import Dataset
from tqdm import tqdm

# from readers.audio import AudioSegment, wav_padding
# from readers.featurizer import wav_slice_padding


def CoughVID_Class(filename="./datasets/waveinfo_labedfine_forcls.csv", isdemo=False):
    print(os.listdir("./datasets"))
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
            # start_time = random.randint(0, len(tmpseg) - 48000)
            start_time = 0
            tmpseg = tmpseg[start_time:start_time + 48000]
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


def min2sec(t: str):
    parts = t.split(':')
    return int(parts[1]) * 60 + float(parts[2])


class CoughVIDReader(object):
    def __init__(self):
        # , ROOT="F:/DATAS/NEUCOUGHDATA_COUGHSINGLE/"
        self.ROOT = "F:/DATAS/COUGHVID-public_dataset_v3/coughvid_20211012_fine/"
        self.metapath = "F:/DATAS/COUGHVID-public_dataset_v3/waveinfo_fewtoml_split.csv"
        # metadf = pd.read_csv(ROOT + "bilicough_metainfo.csv", delimiter=',', header=0,
        #                      index_col=None, encoding="ansi")
        self.m2l = {"healthy": 0, "COVID-19": 1}
        self.sr = None
        self.data_length = None
        self.desc = ""

    def get_sample_label_list(self):
        """
        :param event:
        :return:
            sample_list: sample of cough
            label_list: cough(2)
        """
        sample_list, label_list = [], []
        fin = open(self.metapath, 'r')
        fin.readline()
        line = fin.readline()
        w_data, sr = None, None
        pbar = tqdm(total=2850)
        while line:
            parts = line.split(',')
            fname = parts[1]
            w_data, sr = librosa.load(path=self.ROOT + fname+".wav")
            if self.sr is None:
                self.sr = sr
                self.data_length = sr
                print("coughvid data length:", sr, self.data_length)
            if sr != self.data_length:
                print("Error new sr not equal the data length:", sr, self.data_length)
            segs = None
            if len(w_data) > self.data_length:
                segs = self.__split(w_data)
            elif len(w_data) < self.data_length:
                segs = self.__padding(w_data)
            else:
                segs = [w_data]
            sample_list.extend(segs)
            label_list.extend([2]*len(segs))
            line = fin.readline()
            pbar.update(1)
        fin.close()
        return sample_list, label_list

    def __split(self, w_data):
        L = w_data.shape[0]
        overlap = int(self.data_length // 6)
        if L - self.data_length < overlap:
            st = random.randint(0, L-self.data_length)
            return [w_data[st:st+self.data_length]]
        else:
            segs = []
            st = 0
            while st + self.data_length < L:
                segs.append(w_data[st:st + self.data_length])
                st = st + self.data_length - overlap
            if st + self.data_length - L < overlap:
                segs.extend(self.__padding(w_data[st:]))
            return segs

    def __padding(self, w_data, usenoise=False):
        new_signal = None
        if usenoise:
            pass
        else:
            new_signal = np.zeros(self.data_length)
            # resi = self.data_length - w_data.shape[0]
            # print("resi:", resi)
            new_signal[:w_data.shape[0]] = w_data
            new_signal[-w_data.shape[0]:] = w_data[::-1]
        return [new_signal]


if __name__ == '__main__':
    from torch.utils.data import Dataset, DataLoader
    from chapter2_SEDmodel import CoughDataset
    from readers.noise_reader import load_bilinoise_dataset
    cvr = CoughVIDReader()
    sample_list, label_list = cvr.get_sample_label_list()
    noise_list, _ = load_bilinoise_dataset(NOISE_ROOT="G:/DATAS-Medical/BILINOISE/", noise_length=cvr.data_length,
                                           number=100)
    train_loader = DataLoader(
        CoughDataset(audioseg=sample_list, labellist=label_list, noises=noise_list),
        batch_size=64, shuffle=True)
    for batch_id, (x_wav, y_lab) in tqdm(enumerate(train_loader),
                                         desc="Training "):
        x_wav = x_wav.unsqueeze(1)
        print(x_wav.shape, y_lab.shape)
    # # ext_list()
    # # stat_coughvid()
    # label_list = read_labels_from_csv()
    # print(label_list.shape)
    # from torch.utils.data import DataLoader
    # from ackit.data_utils.collate_fn import collate_fn
    # from ackit.data_utils.featurizer import Wave2Mel
    # plist, llist = CoughVID_Class()
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
