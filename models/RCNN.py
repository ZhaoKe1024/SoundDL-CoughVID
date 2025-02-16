#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/2/16 17:45
# @Author: ZhaoKe
# @File : RCNN.py
# @Software: PyCharm
import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        pool_size = params["pool_size"]
        dropout_rate = params["dropout_rate"]
        nn_cnn2d_filt = params["nb_cnn2d_filt"]
        # rnn_size = params["rnn_size"]
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
        self.rnn_module = nn.GRU(input_size=inp, hidden_size=128, num_layers=self.rnn_layer_num, batch_first=True,
                                 bidirectional=True if self.direction == 2 else False)

        self.sed_module = nn.Sequential()
        for ind in range(len(fnn_size[:-1])):
            inp, oup = fnn_size[ind], fnn_size[ind + 1]
            self.sed_module.append(nn.Linear(inp, oup))
            self.sed_module.append(nn.Dropout(p=dropout_rate))
        self.sed_module.append(nn.Linear(fnn_size[-1], 2))
        self.sed_module.append(nn.Softmax(dim=-1))

    def forward(self, x_mel):
        # out = self.feature_extractor(x_mel)
        x_mel = x_mel.unsqueeze(1).transpose(2, 3)  # [16, 1, 128, 16]
        bs = x_mel.shape
        # print("forward batch_size:", bs)
        for layer in self.feature_extractor:
            out = layer(x_mel)
            # print("CNN layer output:", out.shape)
            x_mel = out
        # x_mel, _ = x_mel.max(dim=-1)
        x_mel = x_mel.transpose(1, 2).reshape(bs[0], bs[2], -1)
        # print("shape of CNN output:", x_mel.shape)
        hidden = self.init_hidden(batch_size=bs[0])
        out, hidden = self.rnn_module(x_mel, hidden)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # print("output shape of lstm:", out.shape)
        out = self.sed_module(out[:, -1, :])
        # print("output of full connected nn:", out.shape)
        return out, hidden

    def init_hidden(self, batch_size):
        # 这个函数写在这里，有一定迷惑性，这个不是模型的一部分，是每次第一个向量没有上下文，在这里捞一个上下文，仅此而已。
        hidden = torch.autograd.Variable(
            torch.zeros(2 * self.direction, batch_size, self.hidden_size, device='cuda'))
        return hidden
