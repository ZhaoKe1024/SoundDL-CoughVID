#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/2/23 16:05
# @Author: ZhaoKe
# @File : chapter3_ADRmodel.py
# @Software: PyCharm
import torch
import torch.nn as nn
from modules.disentangle import AME
from models.conv_vae import ConvVAE


class Classifier(nn.Module):
    def __init__(self, dim_embedding, dim_hidden_classifier, num_target_class):
        super(Classifier, self).__init__()
        self.ext = nn.Sequential(
            nn.Linear(dim_embedding, dim_hidden_classifier),
            nn.BatchNorm1d(dim_hidden_classifier),
            nn.ReLU(),
            # nn.Linear(dim_hidden_classifier, dim_hidden_classifier),
            # nn.BatchNorm1d(dim_hidden_classifier),
            # nn.ReLU(),
        )
        self.cls = nn.Linear(dim_hidden_classifier, num_target_class)

    def forward(self, input_data, fe=False):
        feat = self.ext(input_data)
        if fe:
            return self.cls(feat), feat
        else:
            return self.cls(feat)


class Loss4AGEDR(nn.Module):
    def __init__(self):
        super().__init__()
        self.recon_loss = nn.MSELoss()
        self.categorical_loss = nn.CrossEntropyLoss()


class AGEDR(nn.Module):
    def __init__(self):
        super().__init__()
        self.ame1 = AME(class_num=3, em_dim=self.a1len).to(self.device)
        self.ame2 = AME(class_num=4, em_dim=self.a2len).to(self.device)
        self.vae = ConvVAE(inp_shape=(1, 128, 64), latent_dim=self.latent_dim, flat=True).to(self.device)
        self.classifier = Classifier(dim_embedding=self.latent_dim, dim_hidden_classifier=32,
                                     num_target_class=self.class_num).to(self.device)
        # self.classifier = nn.Linear(in_features=self.latent_dim, out_features=self.class_num).to(self.device)
        self.cls_weight = 2
        self.vae_weight = 0.3

        self.align_weight = 0.0025
        self.kl_attri_weight = 0.01  # noise
        self.kl_latent_weight = 0.0125  # clean
        self.recon_weight = 0.05
