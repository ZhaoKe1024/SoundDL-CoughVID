# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2025-02-13 16:20
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch
import torch.nn as nn
import torchaudio


class TemporalAveragePooling(nn.Module):
    def __init__(self):
        """TAP
        Paper: Multi-Task Learning with High-Order Statistics for X-vector based Text-Independent Speaker Verification
        Link: https://arxiv.org/pdf/1903.12058.pdf
        """
        super(TemporalAveragePooling, self).__init__()

    def forward(self, x):
        """Computes Temporal Average Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels)
        """
        x = x.mean(dim=-1)
        # To be compatable with 2D input
        x = x.flatten(start_dim=1)
        return x


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
        self.td_layer1 = torch.nn.Conv1d(in_channels=input_size, out_channels=hidden_size, dilation=1, kernel_size=5,
                                         stride=1)  # IW-5+1
        length = length - 5 + 1  # 83
        self.bn1 = nn.LayerNorm(length)
        self.td_layer2 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, dilation=2, kernel_size=3,
                                         stride=1, groups=hidden_size)  # IW-4+1
        length = length-4  # 80
        self.bn2 = nn.LayerNorm(length)
        self.td_layer3 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, dilation=3, kernel_size=3,
                                         stride=1, groups=hidden_size)  # IW-6+1
        length = length - 6
        self.bn3 = nn.LayerNorm(length)
        self.td_layer4 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, dilation=1, kernel_size=1,
                                         stride=1, groups=hidden_size)  # IW+1
        length = length
        self.bn4 = nn.LayerNorm(length)
        self.td_layer5 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=channels, dilation=1, kernel_size=1,
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


class VADModel(nn.Module):
    def __init__(self, n_mels=64, d_model=32, nhead=4):
        super().__init__()
        # Mel特征提取
        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, n_fft=2048, hop_length=512, n_mels=n_mels
        )

        # 波形分支（时域特征）
        # self.wave_conv = nn.Sequential(
        #     nn.Conv1d(1, 16, kernel_size=1024, stride=2, padding=2),
        #     nn.BatchNorm1d(16),
        #     nn.SiLU(),
        #     nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm1d(32),
        #     nn.SiLU(),
        # )
        self.wave_conv = TDNN_Extractor()

        # Mel分支（频域特征）
        self.mel_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((None, 32))  # 压缩频率维度
        )

        # # Transformer时序建模
        # encoder_layers = TransformerEncoderLayer(
        #     d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1
        # )
        # self.transformer = TransformerEncoder(encoder_layers, num_layers=4)

        # 分类头
        # self.reduction = nn.Sequential(
        #     nn.Conv1d(512, 32)
        # )
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.classifier = nn.Sequential(
            nn.Linear(512, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (B, 1, 30000) 波形输入
        # 波形分支
        wave_feat = self.wave_conv(x)  # (B, 32, 7500)
        # print("wave_feat shape:", wave_feat.shape)
        wave_feat = wave_feat.permute(0, 2, 1)  # (B, 7500, 32)
        print("wav feat shape:", wave_feat.shape)

        # 提取Mel特征
        mel = self.mel_extractor(x)  # .unsqueeze(1)  # (B, 1, n_mels, T)
        mel = torch.log(mel + 1e-6)  # 对数压缩
        print("mel shape", mel.shape)

        # Mel分支
        mel_feat = self.mel_conv(mel)  # (B, 32, 64, 16)
        print("mel feat shape:", mel_feat.shape)
        mel_feat = mel_feat.permute(0, 3, 1, 2).flatten(2)  # (B, 1024, 32)
        print("mel feat shape:", mel_feat.shape)

        # 特征拼接
        combined = torch.cat([wave_feat, mel_feat], dim=-1)  # (B, 7500+1024, 32)
        print("feat shape:", combined.shape)

        # # Transformer编码
        # src_key_padding_mask = (combined.mean(-1) == 0)  # 动态掩码
        # output = self.transformer(
        #     combined,  # .permute(1, 0, 2),
        #     src_key_padding_mask=src_key_padding_mask
        # )  # (T, B, d_model)
        # print(output.shape)
        # 分类
        output = self.pool(combined.mean(dim=1))
        print(output.shape)
        logits = self.classifier(output)  # (B, 1)
        return logits.squeeze(-1)


if __name__ == '__main__':
    vad_model = VADModel()
    # tdnn = TDNN_Extractor()
    x = torch.rand(size=(16, 1, 22050))
    print(vad_model(x).shape)
    # print(tdnn(x).shape)
