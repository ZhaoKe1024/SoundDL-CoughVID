#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/2/16 17:39
# @Author: ZhaoKe
# @File : tdnncnn.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torchaudio
from modules.extractors import TDNN_Extractor


class WSFNN(nn.Module):
    # Waveform Spectrogram Fused Neural Network
    def __init__(self, n_mels=64, class_num=2, latent_dim=1024):
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
        self.wave_conv = TDNN_Extractor(win_size=1024, hop_length=488, overlap=512, channels=latent_dim)

        # Mel分支（频域特征）
        self.mel_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((None, 32))  # 压缩频率维度
        )
        print("Build 2 Convolutional Layer and 1 Pool2d Layer.")
        # # Transformer时序建模
        # encoder_layers = TransformerEncoderLayer(
        #     d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1
        # )
        # self.transformer = TransformerEncoder(encoder_layers, num_layers=4)

        # 分类头
        # self.reduction = nn.Sequential(
        #     nn.Conv1d(512, 32)
        # )
        print("Pooling after Fusioning the TDNN and CNN.")
        self.pool = nn.MaxPool1d(kernel_size=4)

        self.classifier = nn.Sequential(
            nn.Linear(512, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, class_num)
        )
        print("Build 3-Layer MLP as Classifier for {}-class.".format(class_num))

    def forward(self, x, latent=False):
        # x: (B, 1, 22050) 波形输入
        # 波形分支
        wave_feat = self.wave_conv(x)  # (B, 32, 7500)
        # print("wave_feat shape:", wave_feat.shape)
        wave_feat = wave_feat.permute(0, 2, 1)  # (B, 7500, 32)
        # print("wav feat shape:", wave_feat.shape)  # wav feat shape: torch.Size([32, 32, 1024])

        # 提取Mel特征
        mel = self.mel_extractor(x)  # .unsqueeze(1)  # (B, 1, n_mels, T)
        mel = torch.log(mel + 1e-6)  # 对数压缩
        # print("mel shape", mel.shape)

        # Mel分支
        mel_feat = self.mel_conv(mel)  # (B, 32, 64, 16)
        # print("mel feat shape:", mel_feat.shape)
        mel_feat = mel_feat.permute(0, 3, 1, 2).flatten(2)  # (B, 1024, 32)
        # print("mel feat shape:", mel_feat.shape)  # mel feat shape: torch.Size([32, 32, 1024])

        # 特征拼接
        combined = torch.cat([wave_feat, mel_feat], dim=-1)  # (B, 7500+1024, 32)
        # print("feat shape:", combined.shape)  # feat shape: torch.Size([32, 32, 2048])

        # # Transformer编码
        # src_key_padding_mask = (combined.mean(-1) == 0)  # 动态掩码
        # output = self.transformer(
        #     combined,  # .permute(1, 0, 2),
        #     src_key_padding_mask=src_key_padding_mask
        # )  # (T, B, d_model)
        # print(output.shape)

        # 分类
        output = self.pool(combined.mean(dim=1))
        # print(output.shape)  # torch.Size([32, 512])
        logits = self.classifier(output)  # (B, 1)
        if latent:
            return logits, combined
        else:
            return logits  # .squeeze(-1)


if __name__ == '__main__':
    x = torch.rand(size=(32, 1, 22050))
    m = WSFNN()
    logits, lv = m(x, latent=True)
    print(logits.shape, lv.shape)
