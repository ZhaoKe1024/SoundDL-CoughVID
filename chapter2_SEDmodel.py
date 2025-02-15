#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/11/13 20:57
# @Author: ZhaoKe
# @File : chapter2_SEDmodel.py
# @Software: PyCharm
import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader

from modules.extractors import TDNN_Extractor
from readers.bilicough_reader import BiliCoughReader
from readers.neucough_reader import NEUCoughReader
from readers.coughvid_reader import CoughVIDReader
# from readers.featurizer import Wave2Mel


def get_combined_data():
    bcr = BiliCoughReader()
    ncr = NEUCoughReader()
    cvr = CoughVIDReader()
    sample_list, label_list = [], []
    tmp_sl, tmp_ll = bcr.get_sample_label_list(mode="sed")
    sample_list.extend(tmp_sl)
    label_list.extend(tmp_ll)
    print("bilicough:", len(label_list))
    tmp_sl, tmp_ll = ncr.get_sample_label_list(mode="cough")
    sample_list.extend(tmp_sl)
    label_list.extend(tmp_ll)
    print("bilicough+neucough:", len(label_list))
    tmp_sl, tmp_ll = cvr.get_sample_label_list()
    sample_list.extend(tmp_sl)
    label_list.extend(tmp_ll)
    print("bilicough+neucough+coughvid:", len(label_list))
    return sample_list, label_list


class BiliCoughDataset(Dataset):
    def __init__(self, audioseg, labellist, noises):
        self.audioseg = audioseg
        self.labellist = labellist
        self.noises = noises

    def __getitem__(self, ind):
        # When reading waveform data, add noise as data enhancement according to a 1/3 probability.
        if random.random() < 0.333:
            return self.audioseg[ind] + self.noises[random.randint(0, len(self.noises) - 1)], self.labellist[ind]
        else:
            return self.audioseg[ind], self.labellist[ind]

    def __len__(self):
        return len(self.audioseg)


class SEDModel(nn.Module):
    def __init__(self, n_mels=64, class_num=2):
        super().__init__()
        # Mel特征提取
        self.mel_extractor = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=2048, hop_length=512, n_mels=n_mels)
        self.wave_conv = TDNN_Extractor()

    def forward(self, x_wav):
        return self.wave_conv(x_wav)


class Trainer2SED(object):
    def __init__(self):
        self.configs = {"batch_size": 32, "lr": 0.001, "epoch_num": 30}
        self.save_dir = "./runs/c2vadmodel/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.run_save_dir = self.save_dir + time.strftime("%Y%m%d%H%M", time.localtime()) + '/'
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.save_setting_str = "Model:{}, optimizer:{}, loss function:{}\n".format(
            "VADModel(wav TDNN + mel CNN + pool + mlp)", "Adam(lr={})".format(self.configs["lr"]),
            "nn.CrossEntropyLoss")
        self.save_setting_str += "dataset:{}, batch_size:{}, noise_p:{}\n".format("BiliCough+BiliNoise",
                                                                                  self.configs["batch_size"], "0.333")
        self.save_setting_str += "epoch_num:{},\n".format(self.configs["epoch_num"])

    def __build_model(self):
        self.model = SEDModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.configs["lr"])



if __name__ == '__main__':
    pass
