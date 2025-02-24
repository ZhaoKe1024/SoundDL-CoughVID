#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/2/19 12:31
# @Author: ZhaoKe
# @File : chapter4_SCDE2Emodel.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torchaudio
from modules.extractors import TDNN_Extractor
import torch.nn.functional as F
from modules.attentions import SimpleSelfAttention, SimpleGCNLayer
name2label = {"breathe": 0, "cough": 2, "clearthroat": 1, "exhale": 3, "hum": 4, "inhale": 5, "noise": 6, "silence": 7,
              "sniff": 8, "speech": 9, "vomit": 10, "whooping": 11}
# 最关键保留0，1，2，3，5，10，11，8
# 6，7合并
# 4，9合并


class CSEDNN(nn.Module):
    # Waveform Spectrogram Causal Sound Event Detection Neural Network
    def __init__(self, n_mels=64, class_num=2, latent_dim=1024, fuse_model="attn", hidden_dim=64):
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
        self.wave_conv = TDNN_Extractor(win_size=1024, hop_length=420, overlap=512, channels=latent_dim, wav_len=220500)

        # Mel分支（频域特征）
        self.mel_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((None, self.wave_conv.length))  # 压缩频率维度
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
        self.pool = nn.MaxPool1d(kernel_size=8)

        self.align_weight = 0.0025  # ame
        self.kl_attri_weight = 0.01  # noise
        self.kl_latent_weight = 0.0125  # clean

        self.multi_event_num = 8
        self.event_len = self.wave_conv.length // self.multi_event_num
        self.event_dim = self.wave_conv.length
        self.fuse_model = fuse_model
        self.fuse_layer = None
        n_heads = 4
        if fuse_model == "attn":
            self.fuse_layer = SimpleSelfAttention(d_model=self.event_dim, n_heads=n_heads)
        elif fuse_model == "gnn":
            self.fuse_layer = SimpleGCNLayer(in_channels=self.event_dim, out_channels=hidden_dim)
        else:
            raise ValueError("Unknown fused model:{}(at __init__()).".format(fuse_model))
        # 如果注意力要输出到 hidden_dim，可再加一层线性
        self.attention_fc = nn.Linear(self.event_dim, hidden_dim)
        # 邻接矩阵(若使用 GCN)
        # 简单示例：完全连接图 (不含自环或含自环均可)
        # 通常需要归一化 A 矩阵
        adj = torch.ones(self.multi_event_num, self.multi_event_num)
        # 对角线也可设为1(含自环)，或根据需要设为0
        # adj = adj.fill_diagonal_(0)  # 如果不想要自环，可以这样做
        self.register_buffer('adj', adj)

        print("Pooling after Fusioning the TDNN and CNN.")
        self.pool = nn.MaxPool1d(kernel_size=4)
        # 最终分类器
        self.classifier = nn.Linear(hidden_dim, class_num)
        # 构造一个全连接层以便在两种模式下可以统一输出
        self.dropout = nn.Dropout(p=0.1)

        print("Build 3-Layer MLP as Classifier for {}-class.".format(class_num))

    def forward(self, x):
        # x: (B, 1, 22050) 波形输入
        # 波形分支
        wave_feat = self.wave_conv(x)  # (B, 32, 7500)
        # print("wave_feat shape:", wave_feat.shape)
        wave_feat = wave_feat.permute(0, 2, 1)  # (B, 7500, 32)
        print("wav feat shape:", wave_feat.shape)  # wav feat shape: torch.Size([32, 32, 1024])

        # 提取Mel特征
        mel = self.mel_extractor(x)  # .unsqueeze(1)  # (B, 1, n_mels, T)
        mel = torch.log(mel + 1e-6)  # 对数压缩
        # print("mel shape", mel.shape)

        # Mel分支
        mel_feat = self.mel_conv(mel)  # (B, 32, 64, 16)
        print("mel feat shape:", mel_feat.shape)
        mel_feat = mel_feat.permute(0, 3, 1, 2).flatten(2)  # (B, 1024, 32)
        print("mel feat shape:", mel_feat.shape)  # mel feat shape: torch.Size([32, 32, 1024])

        # 特征拼接
        combined = torch.cat([wave_feat, mel_feat], dim=-1)  # (B, 7500+1024, 32)
        print("feat shape:", combined.shape)  # feat shape: torch.Size([32, 32, 2048])
        #
        # # # Transformer编码
        # # src_key_padding_mask = (combined.mean(-1) == 0)  # 动态掩码
        # # output = self.transformer(
        # #     combined,  # .permute(1, 0, 2),
        # #     src_key_padding_mask=src_key_padding_mask
        # # )  # (T, B, d_model)
        # # print(output.shape)
        #
        # 分类
        bs, _, lv_dim = combined.shape
        combined = combined.view(bs, self.multi_event_num, self.event_len, lv_dim)
        combined = self.pool(combined.mean(dim=2))
        print(combined.shape)  # torch.Size([32, 512])
        print(combined.shape)  # torch.Size([bs, 8, 64, 512])
        print("latent_me:", combined.shape)
        if self.fuse_model == "attn":
            out, attn_weights = self.fuse_layer(combined)
            out = self.attention_fc(out)
            out = F.relu(out)
            out = out.mean(dim=1)
        elif self.fuse_model == "gnn":
            out = self.fuse_layer(combined, self.adj)
            out = F.relu(out)
            out = out.mean(dim=1)
        else:
            raise ValueError("Unknown fused model:{}(at forward().).".format(self.fuse_model))
        print("out shape:", out.shape)
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits


if __name__ == '__main__':
    wav = torch.rand(size=(64, 220500))
    model = CSEDNN(fuse_model="gnn")
    model(wav.unsqueeze(1))
