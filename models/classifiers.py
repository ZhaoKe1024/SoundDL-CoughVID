#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/30 10:10
# @Author: ZhaoKe
# @File : classifiers.py
# @Software: PyCharm
import torch.nn as nn
from modules.attentions import DotAttention, SoftAttention


class ConvEncoder(nn.Module):
    def __init__(self, inp_shape=(1, 298, 512), n_class=3):
        super().__init__()
        c, h, w = inp_shape
        hh, ww = h, w
        self.shapes = [(hh, ww)]
        self.encoder = nn.Sequential()
        cl = [16, 32, -1, 64, 128, -1, 512]
        ksp = [(4, 2, 1), ((5, 4), 2, 1), (2, 2, -1), (4, (1, 2), 1), (4, 2, 1), (2, 2, -1), (3, 2, 0)]
        pre_c = 1
        for i, (k, s, p) in enumerate(ksp):
            if cl[i] == -1:
                self.encoder.append(nn.MaxPool2d(kernel_size=k, stride=s, return_indices=False))
                hh /= 2
                ww /= 2
            else:
                self.encoder.append(nn.Conv2d(pre_c, cl[i], kernel_size=k, stride=s, padding=p))
                self.encoder.append(nn.BatchNorm2d(cl[i]))
                self.encoder.append(nn.ReLU(inplace=True))
                pre_c = cl[i]
                if isinstance(k, tuple):
                    hh = (hh - k[0] + 2 * p) // s + 1
                    ww = (ww - k[1] + 2 * p) // s + 1
                elif isinstance(s, tuple):
                    hh = (hh - k + 2 * p) // s[0] + 1
                    ww = (ww - k + 2 * p) // s[1] + 1
                else:
                    hh = (hh - k + 2 * p) // s + 1
                    ww = (ww - k + 2 * p) // s + 1
            self.shapes.append((hh, ww))
        print(self.shapes)

        self.flatten = nn.Flatten(start_dim=1)
        print("zero later:", cl[-1] * hh * ww)
        hidden_size = [int(cl[-1] * hh * ww), 256, 64, n_class]
        self.cls = nn.Sequential()
        for i in range(len(hidden_size) - 1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i + 1]
            self.cls.append(nn.Linear(in_dim, out_dim))
            if (i < len(hidden_size) - 2):
                self.cls.append(nn.BatchNorm1d(out_dim))
                self.cls.append(nn.ReLU(inplace=True))
            elif (i == len(hidden_size) - 2):
                self.cls.append(nn.BatchNorm1d(out_dim))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_input):
        # feat = self.mp1(self.encoder_conv1(x_input))
        # print(feat.shape)
        # feat = self.mp2(self.encoder_conv2(feat))
        # feat = self.encoder_conv3(feat)
        feat = self.encoder(x_input)
        print("after encoder:", feat.shape)
        feat = self.flatten(feat)
        print("after flatten:", feat.shape)
        feat = self.cls(feat)
        print("after cls:", feat.shape)
        pred = self.softmax(feat)
        print("after softmax:", pred.shape)
        return pred


class LSTM_Classifier(nn.Module):
    def __init__(self, inp_size, hidden_size, n_classes):
        super(LSTM_Classifier, self).__init__()
        self.lstm = nn.LSTM(inp_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x.transpose(1, 2))
        lstm_out = lstm_out[:, -1, :]
        out = self.classifier(lstm_out)
        return out


class LSTM_Attn_Classifier(nn.Module):
    def __init__(self, inp_size, hidden_size, n_classes, return_attn_weights=False, attn_type='dot'):
        super(LSTM_Attn_Classifier, self).__init__()
        self.return_attn_weights = return_attn_weights
        self.lstm = nn.LSTM(inp_size, hidden_size, batch_first=True)
        self.attn_type = attn_type

        if self.attn_type == 'dot':
            self.attention = DotAttention()
        elif self.attn_type == 'soft':
            self.attention = SoftAttention(hidden_size, hidden_size)

        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x.transpose(1, 2))

        if self.attn_type == 'dot':
            attn_output = self.attention(lstm_out, hidden)
            attn_weights = self.attention._get_weights(lstm_out, hidden)
        elif self.attn_type == 'soft':
            attn_output = self.attention(lstm_out)
            attn_weights = self.attention._get_weights(lstm_out)

        out = self.classifier(attn_output)
        if self.return_attn_weights:
            return out, attn_weights
        else:
            return out


def test_lstm():
    import torch
    # input_size: 时间步
    # hidden_size:
    # num_layer: 层数
    x = torch.randn(64, 64, 128)  # (bs, length, dim)
    lstm = LSTMClassifier(input_size=128, hidden_size=128, num_layers=2, num_classes=2)
    print(lstm(x))


if __name__ == '__main__':
    main()
    # test_lstm()