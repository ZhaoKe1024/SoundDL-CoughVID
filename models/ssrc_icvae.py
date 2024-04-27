# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-04-27 22:37
import torch
from torch import nn
from models.conv_vae import VEncoder, VDecoder, MLP
"""
Semi-Supervised Framework
Causal-based Classification Temporal-Masked VAE
Can I try to Attributes based Classification for Self-Supervised?
"""


class ICVAE(nn.Module):
    def __init__(self, inp_shape=(1, 94, 128), class_num=3, latent_dim=128, feat_c=8, causal_dim=32, flat=True):
        super().__init__()
        self.flat = flat
        self.encoder = VEncoder(inp_shape=inp_shape, flat=flat)
        self.cc, hh, ww = self.encoder.cc, self.encoder.hh, self.encoder.ww
        if flat:
            self.calc_mean = MLP([self.cc * hh * ww, 128, 64, latent_dim], last_activation=False)
            self.calc_logvar = MLP([self.cc * hh * ww, 128, 64, latent_dim], last_activation=False)
        else:
            self.calc_mean = nn.Conv2d(128, feat_c, kernel_size=1, stride=1, padding=0,
                                       bias=False)
            self.calc_logvar = nn.Conv2d(128, feat_c, kernel_size=1, stride=1, padding=0,
                                         bias=False)
        self.decoder = VDecoder(inp_shape=(self.cc, hh, ww), flat=flat, latent_dim=latent_dim, feat_c=feat_c)


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    enc1 = VEncoder(inp_shape=(1, 94, 128), flat=True).to(device)
    enc2 = VEncoder(inp_shape=(1, 94, 128), flat=False).to(device)

    # x = torch.randn(size=(16, 1, 94, 128)).to(device)
    # x_recon, z, mean_lant, logvar_lant = m(x)
    # print(x_recon.shape, z.shape, mean_lant.shape, logvar_lant.shape)

    # dec = VDecoder(inp_shape=(1, 94, 128), flat=True).to(device)
