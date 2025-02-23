#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/2/23 16:05
# @Author: ZhaoKe
# @File : disentangle.py
# @Software: PyCharm
import torch
import torch.nn as nn


class AME(nn.Module):
    def __init__(self, em_dim, class_num=2, oup=16, mode="dis"):
        super().__init__()
        layers = []
        if mode == "dis":
            layers.extend([nn.Embedding(num_embeddings=class_num, embedding_dim=oup)])
        elif mode == "con":
            pass
            # layers.extend([nn.Embedding(num_embeddings=class_num, embedding_dim=oup)])
        else:
            raise ValueError("Error AME mode {}, please choose from [\"dis\", \"con\"].".format(mode))
        layers.extend([nn.Linear(in_features=oup, out_features=oup), nn.LeakyReLU()])
        layers.extend([nn.Linear(in_features=oup, out_features=oup), nn.LeakyReLU()])
        self.net = nn.Sequential(*layers)
        self.emb_lin_mu = nn.Linear(oup, em_dim)
        self.emb_lin_lv = nn.Linear(oup, em_dim)

    @staticmethod
    def sampling(mean, logvar, device=torch.device("cuda")):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, in_a, mu_only=False):
        """

        :param mu_only:
        :param in_a:
        :return: mu, logvar, z
        """
        mapd = self.net(in_a)
        res_mu = self.emb_lin_mu(mapd)
        if mu_only:
            return res_mu
        else:
            res_logvar = self.emb_lin_lv(mapd)
            return res_mu, res_logvar, self.sampling(res_mu, res_logvar)
