#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/2/15 20:35
# @Author: ZhaoKe
# @File : extractors.py
# @Software: PyCharm
import torch.nn as nn


class TDNN_Extractor(nn.Module):
    def __init__(self, num_class=2, input_size=128, hidden_size=512, channels=1024, embd_dim=32):
        super(TDNN_Extractor, self).__init__()
        self.emb_size = embd_dim
        kernel_size, stride, padding = 1024, 488, 512
        self.wav2mel = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=kernel_size, stride=stride,
                                 padding=padding, bias=False)
        length = (22050 + 2 * padding - kernel_size) // stride + 1  # 87, who to 46?
        self.layer_norm = nn.LayerNorm(length)
        # self.wav2mel = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=1024, stride=512, padding=1024 // 2, bias=False)
        self.td_layer1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, dilation=1, kernel_size=5,
                                   stride=1)  # IW-5+1
        length = length - 5 + 1  # 83
        self.bn1 = nn.LayerNorm(length)
        self.td_layer2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, dilation=2, kernel_size=3,
                                   stride=1, groups=hidden_size)  # IW-4+1
        length = length - 4  # 80
        self.bn2 = nn.LayerNorm(length)
        self.td_layer3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, dilation=3, kernel_size=3,
                                   stride=1, groups=hidden_size)  # IW-6+1
        length = length - 6
        self.bn3 = nn.LayerNorm(length)
        self.td_layer4 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, dilation=1, kernel_size=1,
                                   stride=1, groups=hidden_size)  # IW+1
        length = length
        self.bn4 = nn.LayerNorm(length)
        self.td_layer5 = nn.Conv1d(in_channels=hidden_size, out_channels=channels, dilation=1, kernel_size=1,
                                   stride=1, groups=hidden_size)  # IW+1
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

        # self.pooling = TemporalAveragePooling()
        # self.bn5 = nn.BatchNorm1d(channels)
        # self.linear = nn.Linear(channels, embd_dim)
        # self.bn6 = nn.BatchNorm1d(embd_dim)
        # self.fc = nn.Linear(embd_dim, num_class)

    def forward(self, waveform):
        # x = x.transpose(2, 1)
        tgram = self.wav2mel(waveform)
        # print(tgram.shape)
        x = self.td_layer1(self.leakyrelu(self.layer_norm(tgram)))
        # print("shape of x as a wave:", x.shape)
        x = self.td_layer2(self.leakyrelu(self.bn1(x)))
        # print("shape of x in layer 1:", x.shape)
        x = self.td_layer3(self.leakyrelu(self.bn2(x)))
        # print("shape of x in layer 2:", x.shape)
        x = self.td_layer4(self.leakyrelu(self.bn3(x)))
        # print("shape of x in layer 3:", x.shape)
        out = self.td_layer5(self.leakyrelu(self.bn4(x)))
        # print("shape of x in layer 4:", out.shape)
        # out = self.bn5(self.pooling(x))
        # print("shape of x after pooling:", out.shape)
        # out = self.bn6(self.linear(out))
        # out = self.fc(out)
        return out
