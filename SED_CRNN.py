#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/11/13 20:57
# @Author: ZhaoKe
# @File : SED_CRNN.py
# @Software: PyCharm


"""
def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size,
                                rnn_size, fnn_size, classification_mode, weights):
    # model definition
    spec_start = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))
    spec_cnn = spec_start
    for i, convCnt in enumerate(pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
    spec_cnn = Permute((2, 1, 3))(spec_cnn)

    spec_rnn = Reshape((data_in[-2], -1))(spec_cnn)
    for nb_rnn_filt in rnn_size:
        spec_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)

    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)

    doa = TimeDistributed(Dense(data_out[1][-1]))(doa)
    doa = Activation('tanh', name='doa_out')(doa)

    # SED
    sed = spec_rnn
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    model = Model(inputs=spec_start, outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)

    model.summary()
    return model
"""
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import librosa
import torch
from torch import nn
from readers.featurizer import Wave2Mel

ROOT = "F:/DATAS/NEUCOUGHDATA_COUGHSINGLE/"


def min2sec(t: str):
    parts = t.split(':')
    return int(parts[0]) * 60 + float(parts[1])


def get_rand_start(sec, wav_length: int, mode="left", full_length: int = None, sr=22050):
    """ 除法/不要写成//，会导致算什么都是0 """
    if mode == "left":
        return np.random.randint(0, sec * sr - wav_length) / sr
    elif mode == "right":
        return np.random.randint(sec * sr, full_length - wav_length) / sr
    else:
        raise Exception("Unknown mode of get_rand_start(mode=\"???\").")


def read_audio(filepath: str, st=None, en=None, duration=None):
    # print(y.shape, sr)
    # sr = 22050
    if st is not None:
        if en is not None:
            # st, en = int(st * sr), int(en * sr)
            # print("st, en:", st, en)
            y, sr = librosa.load(filepath, offset=st, duration=en - st)
            # print("y, sr:", len(y), sr)
        elif duration is not None:
            y, sr = librosa.load(filepath, offset=st, duration=duration)
        else:
            raise Exception("Arguments Error of read_audio()")
    else:
        y, sr = librosa.load(filepath)
        # print("y, sr:", y, sr)
    # print(y.shape)
    # mel = w2m(torch.from_numpy(y))
    return y, sr


def labelling_by_IOU(st1, en1, st2, d2, mode="left"):
    # print(st1, en1, st2, d2)
    if mode == "left":
        if st2 + d2 < st1:
            return 0  # other
        elif (st2 + d2 - st1) / (en1 - st2) > 0.2:
            return 1
        else:
            return 0
    else:
        raise Exception("Unknown mode of labelling_by_IOU(mode=\"???\")")

def wav_crop_padding(mel, fixed_length=16):
    dim, length = mel.shape
    if length < 16:
        new_mel = np.zeros((128, 16))
        st = np.random.randint(0, (fixed_length - length + 1) // 2)
        new_mel[:, st:st + 16] = mel
    elif length > 16:
        st = np.random.randint(0, (length + 1 - fixed_length) // 2)
        new_mel = mel[:, st:st + 16]
    else:
        new_mel = mel
    return new_mel

def read_metafile(w2m):
    slice_raw = pd.read_csv(ROOT + 'neucough_metainfo_slice.txt', header=0, index_col=0)
    lab2name = {0: "non-cough", 1: "cough"}
    fixed_length = 16

    for idx, item in enumerate(slice_raw.itertuples()):
        name, st, en = item[0], item[1], item[2]
        st, en = min2sec(st), min2sec(en)
        print("st, en:", st, en)
        audio_path = "F:/DATAS/NEUCOUGHDATA_FULL/{}_audiodata_元音字母a.wav".format(name)
        pos_sample, sr = read_audio(audio_path, st=st, en=en)
        pos_mel = w2m(torch.from_numpy(pos_sample).to(torch.float32))
        # 有一个很大的问题，正样本都是填充0更敏感，负样本却是随机截取的更鲁棒，要想办法也把正样本也调整st, en来随机截取
        pos_mel = wav_crop_padding(pos_mel, fixed_length=fixed_length)
        print("pos_sample:", pos_sample.shape, pos_mel.shape)
        print("label:", 1, lab2name[1])

        mel16_rand_wavlen = np.random.randint(7680, 8192)  # Mel length: 16
        print("wavlen:", mel16_rand_wavlen / sr)
        rand_start = get_rand_start(sec=st, wav_length=mel16_rand_wavlen, mode="left", full_length=None, sr=22050)
        print("rand_start:", rand_start)
        neg_sample, _ = read_audio(audio_path, st=rand_start, duration=mel16_rand_wavlen / sr)  # 除法/不要写成//，会导致算什么都是0
        neg_mel = w2m(torch.from_numpy(neg_sample).to(torch.float32))
        print("neg_sample:", neg_sample.shape, neg_mel.shape)
        gen_label = labelling_by_IOU(st1=st, en1=en, st2=rand_start, d2=mel16_rand_wavlen / sr)
        print("label:", gen_label, lab2name[gen_label])
        while gen_label == 1:
            mel16_rand_wavlen = np.random.randint(7680, 8192)  # Mel length: 16
            print("wavlen:", mel16_rand_wavlen / sr)
            rand_start = get_rand_start(sec=st, wav_length=mel16_rand_wavlen, mode="left", full_length=None, sr=22050)
            print("rand_start:", rand_start)
            neg_sample, _ = read_audio(audio_path, st=rand_start,
                                       duration=mel16_rand_wavlen / sr)  # 除法/不要写成//，会导致算什么都是0
            print("neg_sample:", neg_sample.shape)
            gen_label = labelling_by_IOU(st1=st, en1=en, st2=rand_start, d2=mel16_rand_wavlen / sr)
            print("label:", gen_label, lab2name[gen_label])
        print()
        if idx > 2:
            print("idx:", idx, mel16_rand_wavlen)
            break


class CRNN(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        pool_size = params["pool_size"]
        dropout_rate = params["dropout_rate"]
        nn_cnn2d_filt = params["nb_cnn2d_filt"]
        rnn_size = params["rnn_size"]
        fnn_size = params["fnn_size"]
        inp, oup = 1, nn_cnn2d_filt
        self.feature_extractor = nn.Sequential()
        print("Build CRNN Model:")
        pool_cnt = 2
        for idx, convCnt in enumerate(pool_size):
            print("Layer:", idx)
            self.feature_extractor.append(nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=(3, 3), padding=1))
            self.feature_extractor.append(nn.BatchNorm2d(num_features=oup))
            self.feature_extractor.append(nn.ReLU())
            if pool_cnt > 0:
                self.feature_extractor.append(nn.MaxPool2d((1, convCnt)))
                self.feature_extractor.append(nn.Dropout(p=dropout_rate))
                pool_cnt -= 1
            inp, oup = oup, 64

        # rnn_module = nn.Sequential()
        # for nb in rnn_size:
        #     rnn_module.append(nn.GRU())

        inp = 128
        self.hidden_size = 128
        self.direction = 2
        self.rnn_layer_num = 2
        self.rnn_module = torch.nn.GRU(input_size=inp, hidden_size=128, num_layers=self.rnn_layer_num, batch_first=True,
                                       bidirectional=True if self.direction == 2 else False)

        self.sed_module = nn.Sequential()
        for ind in range(len(fnn_size[:-1])):
            inp, oup = fnn_size[ind], fnn_size[ind + 1]
            self.sed_module.append(nn.Linear(inp, oup))
            self.sed_module.append(nn.Dropout(p=dropout_rate))
        self.sed_module.append(nn.Linear(fnn_size[-1], 2))
        self.sed_module.append(nn.Softmax())

    def forward(self, x_mel):
        # out = self.feature_extractor(x_mel)
        bs = x_mel.shape
        print(bs)
        for layer in self.feature_extractor:
            out = layer(x_mel)
            print("CNN layer output:", out.shape)
            x_mel = out
        x_mel = x_mel.squeeze().transpose(1, 2).reshape(bs[0], bs[2], -1)
        print("shape of CNN output:", x_mel.shape)
        hidden = self.init_hidden(batch_size=bs[0])
        out, hidden = self.rnn_module(x_mel, hidden)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        print("output shape of lstm:", out.shape)
        out = self.sed_module(out[:, -1, :])
        print("output of full connected nn:", out.shape)
        return out, hidden

    def init_hidden(self, batch_size):
        # 这个函数写在这里，有一定迷惑性，这个不是模型的一部分，是每次第一个向量没有上下文，在这里捞一个上下文，仅此而已。
        hidden = torch.autograd.Variable(
            torch.zeros(2 * self.direction, batch_size, self.hidden_size, device='cpu'))
        return hidden


def main():
    # pool_size = [8, 8, 2]
    params = {"pool_size": [8, 8, 2], "dropout_rate": 0.0, "batch_size": 32, "nb_cnn2d_filt": 64,
              "rnn_size": [128, 128], "fnn_size": [256, 128, 32]}
    crnn_model = CRNN(params=params)
    print(crnn_model)
    x = torch.rand(size=(32, 1, 16, 128))
    crnn_model(x)
    # # loss_f1 = nn.CrossEntropyLoss()
    # # loss_f2 = nn.MSELoss()
    # # optim = torch.optim.Adam()
    # print(crnn_model)


if __name__ == '__main__':
    main()
