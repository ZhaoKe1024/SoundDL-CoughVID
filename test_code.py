#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/12/4 15:54
# @Author: ZhaoKe
# @File : test_code.py
# @Software: PyCharm
"""
暂时测试代码用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def redundancy_overlap_generate():
    length = 46
    data_length = 11
    cnt_sum = length // data_length + 1
    res = cnt_sum * data_length - length
    print(cnt_sum, res)
    overlap = res // (cnt_sum - 1)
    print(overlap)
    st = 0
    while st + data_length <= length:
        print("[{}, {}]".format(st, st + data_length))
        st += data_length - overlap
    print("[{}, {}]".format(length - data_length - 1, length - 1))


def min2sec(t: str):
    parts = t.split(':')
    res = float(parts[-1])
    f = 60
    for i in range(len(parts) - 1):
        res += int(parts[len(parts) - 2 - i]) * f
        f *= 60
    return res


class Wave2Mel(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        return self.amplitude_to_db(self.mel_transform(x))


def find_wavlength():
    w2m = Wave2Mel(sr=22050)
    for length in range(22050, 33075, 1000):
        print(w2m(torch.rand(4, length)).shape)


class TDNN(nn.Module):
    def __init__(self, window=1024, overlap=768):
        super(TDNN, self).__init__()
        self.tdnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=window, stride=window - overlap, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=1)
        )

    def forward(self, x):
        return self.tdnn(x).squeeze()  # .max(dim=-1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(4, 2), stride=(2, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=(4, 2), stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=(4, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 3), stride=1),
        )

    def forward(self, x):
        return self.cnn(x).squeeze()  # .max(dim=-1, ).max(dim=-1)


class AudioClassifier(nn.Module):
    def __init__(self, fused_hidden_dim=64, num_classes=2):
        super(AudioClassifier, self).__init__()

        # 时域特征提取模块 (TDNN)
        self.tdnn_module = TDNN()

        # 频域特征提取模块 (CNN)
        self.cnn_module = CNN()

        # 融合与分类:
        # 1) 先将 TDNN 的输出转换为向量
        # 2) CNN 的输出也转换为向量
        # 3) 拼接后再做一个全连接映射到 latent vector
        # 4) 最后再接一个分类层

        # 这里我们需要计算或指定 CNN 输出的特征维度 (flatten 之后)，
        # 以及 TDNN 输出的特征维度 (flatten 之后)。
        # 这两个数和输入的形状、网络结构相关，需要事先推断或手动指定。
        # 这里写一个示例，假设输入大小固定或我们已经测量过。

        # 假设 TDNN 输出形状: [batch_size, tdnn_out_channels, time_length_out]
        # 这里我们在 forward 里根据实际 shape 做自适应 flatten。

        # 假设 CNN 输出形状: [batch_size, cnn_num_filters*2, freq_out, time_out]
        # 同理在 forward 里 flatten。

        # 先定义一个融合后的全连接层：从  (tdnn_dim + cnn_dim) -> fused_hidden_dim
        self.fusion_fc = nn.Linear(
            in_features=128,  # 这里假设我们在时间/频率上进一步聚合
            out_features=fused_hidden_dim
        )

        # 最终分类层
        self.classifier = nn.Linear(fused_hidden_dim, num_classes)

    def forward(self, x_time, x_freq):
        """
        x_time: [batch_size, tdnn_in_channels, time_length]
        x_freq: [batch_size, 1, freq_bins, time_frames]
        """
        # 1) 时域特征提取
        tdnn_out = self.tdnn_module(x_time)
        # tdnn_out: [batch_size, tdnn_out_channels, time_length_out]
        # 这里可以对时间维度做 pooling 或者直接取最后一帧，也可 global average pooling
        # 例如，这里做一个 global average pooling:
        tdnn_out = torch.mean(tdnn_out, dim=2)  # [batch_size, tdnn_out_channels]

        # 2) 频域特征提取
        cnn_out = self.cnn_module(x_freq)
        # cnn_out: [batch_size, cnn_num_filters*2, freq_out, time_out]
        # 同理，这里也做一个 global average pooling
        cnn_out = torch.mean(cnn_out, dim=[2, 3])  # [batch_size, cnn_num_filters*2]

        # 3) 将两者融合 (拼接)
        fused = torch.cat((tdnn_out, cnn_out), dim=1)  # [batch_size, tdnn_out_channels + cnn_num_filters*2]

        # 4) 融合后映射到隐向量 (latent vector)
        fused_hidden = F.relu(self.fusion_fc(fused))  # [batch_size, fused_hidden_dim]

        # 5) 得到分类结果
        logits = self.classifier(fused_hidden)  # [batch_size, num_classes]

        return logits


def model_design_task1():
    tdnn_module = TDNN(window=1024, overlap=768)
    cnn_module = CNN()
    cls_module = AudioClassifier()

    x_wav = torch.rand(size=(32, 22050))
    x_mel = torch.rand(size=(32, 128, 44))
    tfm = tdnn_module(x_wav.unsqueeze(1))
    ffm = cnn_module(x_mel.unsqueeze(1))
    fm = torch.concat([tfm, ffm], dim=-1)
    print("feature map:", fm.shape)



if __name__ == '__main__':
    # redundancy_overlap_generate()
    # print(min2sec("00:01:19.06")*22050)
    # find_wavlength()
    model_design_task1()
