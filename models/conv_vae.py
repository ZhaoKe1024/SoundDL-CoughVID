#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/25 16:03
# @Author: ZhaoKe
# @File : conv_vae.py
# @Software: PyCharm
from collections import OrderedDict
import torch
from torch import nn


class ConvVAE(nn.Module):
    def __init__(self, shape=(1, 94, 128), hidden_dim=128, flat=True):
        super().__init__()
        c, h, w = shape
        hh, ww = h, w
        self.shapes = [(hh, ww)]

        self.encoder_conv1 = nn.Sequential()

        self.encoder_conv1.append(nn.Conv2d(c, 16, kernel_size=4, stride=2, padding=1))
        self.encoder_conv1.append(nn.BatchNorm2d(16))
        self.encoder_conv1.append(nn.ReLU(inplace=True))
        hh = int((hh - 4 + 2 * 1) / 2) + 1  # 47
        ww = int((ww - 4 + 2 * 1) / 2) + 1  # 64
        self.shapes.append((hh, ww))

        self.encoder_conv1.append(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1))
        self.encoder_conv1.append(nn.BatchNorm2d(32))
        self.encoder_conv1.append(nn.ReLU(inplace=True))
        hh = int((hh - 3 + 2 * 1) / 2) + 1  # 24
        ww = int((ww - 3 + 2 * 1) / 2) + 1  # 32
        self.shapes.append((hh, ww))

        # self.maxpool2d1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # hh, ww = hh // 2, ww // 2
        # self.shapes.append(())

        self.encoder_conv2 = nn.Sequential()
        self.encoder_conv2.append(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1))
        self.encoder_conv2.append(nn.BatchNorm2d(64))
        self.encoder_conv2.append(nn.ReLU(inplace=True))
        hh = int((hh - 4 + 2 * 1) / 2) + 1  # 12
        ww = int((ww - 4 + 2 * 1) / 2) + 1  # 16
        self.shapes.append((hh, ww))

        self.encoder_conv2.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0))
        self.encoder_conv2.append(nn.BatchNorm2d(128))
        self.encoder_conv2.append(nn.ReLU(inplace=True))
        hh = (hh - 3 + 2 * 0) // 2 + 1  # 5
        ww = (ww - 3 + 2 * 0) // 2 + 1  # 7
        self.shapes.append((hh, ww))
        # -------------------------sample------------
        self.cc = 128
        self.flat = flat
        if flat:
            self.hh, self.ww = self.shapes[-1]
            self.flatten = nn.Flatten(start_dim=1)

            self.calc_mean = MLP([self.cc * hh * ww, 128, 64, hidden_dim], last_activation=False)
            self.calc_logvar = MLP([self.cc * hh * ww, 128, 64, hidden_dim], last_activation=False)

            self.decoder_lin = MLP([hidden_dim, 64, 128, self.cc * self.hh * self.ww])
            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(self.cc, self.hh, self.ww))
        else:

            self.calc_mean = nn.Conv2d(128, 8, kernel_size=1, stride=1, padding=0,
                                       bias=False)
            self.calc_logvar = nn.Conv2d(128, 8, kernel_size=1, stride=1, padding=0,
                                         bias=False)
            self.decoder_proj = nn.Conv2d(8, 128, kernel_size=1, stride=1, padding=0,
                                          bias=False)
        # ------------------------decoder---------------
        self.decoder_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0)
        self.decoder_norm1 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.decoder_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decoder_norm2 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.decoder_conv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.decoder_norm3 = nn.Sequential(nn.BatchNorm2d(16), nn.ReLU(inplace=True))

        self.decoder_conv4 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        # self.cls = nn.Sequential()
        # self.cls.append(nn.Linear(hidden_dim, 32))
        # self.cls.append(nn.BatchNorm1d(32))
        # self.cls.append(nn.ReLU(inplace=True))
        # self.cls.append(nn.Linear(32, class_num))

    def sampling(self, mean, logvar, device=torch.device("cuda")):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x):
        # encoder
        x_feat = self.encoder_conv2(self.encoder_conv1(x))
        # print("x_feat conv2:", x_feat.shape)
        # flatten
        if self.flat:
            x_feat = self.flatten(x_feat)
            # print("x_feat flatten:", x_feat.shape)
            # sample
            mean_lant, logvar_lant = self.calc_mean(x_feat), self.calc_logvar(x_feat)
            z = self.sampling(mean_lant, logvar_lant, device=torch.device("cuda"))
            # unflatten
            x_recon = self.unflatten(self.decoder_lin(z))
        else:
            mean_lant, logvar_lant = self.calc_mean(x_feat), self.calc_logvar(x_feat)
            z = self.sampling(mean_lant, logvar_lant, device=torch.device("cuda"))
            x_recon = self.decoder_proj(z)
        # print("recon:", x_recon.shape)
        # decoder
        x_recon = self.decoder_conv1(x_recon, output_size=self.shapes[-2])
        x_recon = self.decoder_norm1(x_recon)
        x_recon = self.decoder_conv2(x_recon, output_size=self.shapes[-3])
        x_recon = self.decoder_norm2(x_recon)
        x_recon = self.decoder_conv3(x_recon, output_size=self.shapes[-4])
        x_recon = self.decoder_norm3(x_recon)
        x_recon = self.decoder_conv4(x_recon, output_size=self.shapes[-5])
        # x_recon = self.decoder_norm1(x_recon)
        # x_pred = self.cls(x_feat)
        return x_recon, z, mean_lant, logvar_lant


class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation=True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size) - 1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i + 1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size) - 2) or ((i == len(hidden_size) - 2) and last_activation):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))

    def forward(self, x):
        return self.mlp(x)


MSE_loss = nn.MSELoss(reduction="mean")


def vae_loss(X, X_hat, mean, logvar, kl_weight=0.0001):
    reconstruction_loss = MSE_loss(X_hat, X)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean ** 2)
    # print(reconstruction_loss.item(), KL_divergence.item())
    return reconstruction_loss + kl_weight * KL_divergence


if __name__ == '__main__':
    # device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # m = ConvVAE(shape=(1, 94, 128), flat=True).to(device)
    # x = torch.randn(size=(16, 1, 94, 128)).to(device)
    # y_pred = m(x)  # (28-3+2p)/2+1
    # print(y_pred.shape)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    m = ConvVAE(shape=(1, 94, 128), flat=False).to(device)
    x = torch.randn(size=(16, 1, 94, 128)).to(device)
    y_pred = m(x)  # (28-3+2p)/2+1
    print(y_pred.shape)
