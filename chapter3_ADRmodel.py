#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/2/23 16:05
# @Author: ZhaoKe
# @File : chapter3_ADRmodel.py
# @Software: PyCharm
import itertools
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.conv_vae import ConvVAE
from modules.disentangle import AME
from modules.loss import vae_loss_fn, kl_2normal, pairwise_kl_loss


class CoughDataset(Dataset):
    def __init__(self, audioseg, labellist, attri1_list, attri2_list, noises):
        self.audioseg = audioseg
        self.labellist = labellist
        self.noises = noises

    def __getitem__(self, ind):
        # When reading waveform data, add noise as data enhancement according to a 1/3 probability.
        if random.random() < 0.333:
            return self.audioseg[ind] + self.noises[random.randint(0, len(self.noises) - 1)], self.labellist[ind]
        else:
            return self.audioseg[ind], self.labellist[ind]

    def __len__(self):
        return len(self.audioseg)


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


class AGEDR(nn.Module):
    def __init__(self, latent_dim, a1len, a2len, class_num):
        super().__init__()
        self.ame1 = AME(class_num=3, em_dim=a1len)
        self.ame2 = AME(class_num=4, em_dim=a2len)
        self.vae = ConvVAE(inp_shape=(1, 128, 64), latent_dim=latent_dim, flat=True)
        self.classifier = Classifier(dim_embedding=latent_dim, dim_hidden_classifier=32,
                                     num_target_class=class_num)
        self.latent_dim = 30
        self.a1len, self.a2len = a1len, a2len
        self.blen = self.latent_dim - self.a1len - self.a2len
        # self.classifier = nn.Linear(in_features=self.latent_dim, out_features=self.class_num).to(self.device)
        self.vae_weight = 0.3
        self.cls_weight = 2
        self.align_weight = 0.0025
        self.kl_attri_weight = 0.01  # noise
        self.kl_latent_weight = 0.0125  # clean
        self.recon_weight = 0.05
        self.recon_loss = nn.MSELoss()
        self.categorical_loss = nn.CrossEntropyLoss()

    def forward(self, x_input, attri1, attri2, y_lab):
        # ---------------------Loss Vae------------------------
        x_recon, z_mu, z_logvar, z_latent = self.vae(x_input)  # [32, 1, 64, 128] [32, 30] [32, 30] [32, 30]
        Loss_vae = 0.01 * self.vae_weight * vae_loss_fn(recon_x=x_recon, x=x_input, mean=z_mu, log_var=z_logvar)

        # Loss_attri *= self.kl_attri_weight
        # print("shape of attri1 latent:", mu_a_1.shape, logvar_a_1.shape)
        # print("shape of attri2 latent:", mu_a_2.shape, logvar_a_2.shape)
        # print("shape of vae output:", x_recon.shape, z_mu.shape, z_logvar.shape, z_latent.shape)
        # print("shape of y_pred:", y_pred.shape)
        # --------------------Loss attri align--------------------------
        mu_a_1, logvar_a_1, _ = self.ame1(attri1)  # [32, 6] [32, 6]
        mu_a_2, logvar_a_2, _ = self.ame2(attri2)  # [32, 8] [32, 8]

        mu1_latent = z_mu[:, self.blen:self.blen + self.a1len]  # Size([32, 6])
        mu2_latent = z_mu[:, self.blen + self.a1len:self.blen + self.a1len + self.a2len]  # Size([32, 6])
        lv1_latent = z_logvar[:, self.blen:self.blen + self.a1len]  # Size([32, 8])
        lv2_latent = z_logvar[:, self.blen + self.a1len:self.blen + self.a1len + self.a2len]  # Size([32, 8])
        Loss_attri = kl_2normal(mu_a_1, logvar_a_1, mu1_latent, lv1_latent)
        Loss_attri += kl_2normal(mu_a_2, logvar_a_2, mu2_latent, lv2_latent)
        Loss_attri = self.align_weight * Loss_attri

        # -----------------------------Loss Disen--------------------
        bs = len(z_mu)
        Loss_akl = self.kl_latent_weight * pairwise_kl_loss(z_mu[:, :self.blen], z_logvar[:, :self.blen], bs)
        Loss_akl += self.kl_attri_weight * pairwise_kl_loss(z_mu[:, self.blen:], z_logvar[:, self.blen:], bs)
        Loss_akl = Loss_akl.sum(-1)
        Loss_recon = self.recon_weight * self.recon_loss(x_recon, x_input)
        Loss_disen = Loss_akl + Loss_recon

        y_pred = self.classifier(z_mu)  # torch.Size([32, 2])
        # Loss_cls = self.cls_weight * self.categorical_loss(y_pred, y_lab)
        Loss_cls = self.cls_weight * self.focal_loss(y_pred, y_lab)

        return Loss_vae, Loss_attri, Loss_disen, Loss_cls


class ADRTrainer(object):
    def __init__(self):
        self.lambda_cat = 1.
        self.lambda_con = 0.1
        self.img_size, self.channel = (64, 128), 1
        self.class_num, self.batch_size = 2, 16
        self.latent_dim = 30
        self.a1len, self.a2len = 6, 8
        self.blen = self.latent_dim - self.a1len - self.a2len
        self.configs = {
            "recon_weight": 0.01,
            "cls_weight": 1.,
            "kl_beta_alpha_weight": 0.01,
            "kl_c_weight": 0.075,
            "channels": 1,
            "class_num": 10,
            "code_dim": 2,
            "img_size": 32,
            "latent_dim": 62,
            "sample_interval": 400,
            "fit": {
                "b1": 0.5,
                "b2": 0.999,
                "batch_size": 64,
                "epochs": 40,
                "learning_rate": 0.0002,
            }
        }
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __build_dataloader(self):
        pass
        test_batch = [torch.rand(size=(1, 22050)) for _ in range(512)]
        test_y = [torch.randint(low=0, high=2, size=(1,)) for _ in range(512)]
        noise_list = [torch.rand(size=(1, 22050)) for _ in range(512)]
        self.test_loader = DataLoader(
            CoughDataset(audioseg=test_batch, labellist=test_y, noises=noise_list),
            batch_size=self.configs["fit"]["batch_size"], shuffle=True)

    def __build_model(self):
        self.agedr = AGEDR(latent_dim=self.latent_dim, a1len=self.a1len, a2len=self.a2len, class_num=self.class_num).to(
            self.device)
        self.optimizer_Em = torch.optim.Adam(
            itertools.chain(self.agedr.ame1.parameters(), self.agedr.ame2.parameters()), lr=0.0003, betas=(0.5, 0.999))
        self.optimizer_vae = torch.optim.Adam(self.agedr.vae.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.optimizer_cls = torch.optim.Adam(self.agedr.classifier.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train(self):
        pass

        pbar = tqdm(total=self.configs["fit"]["epochs"])
        flag = True
        for epoch_id in range(self.configs["fit"]["epochs"]):
            pbar.set_description(desc="Epoch {}:".format(epoch_id))
            for batch_id, (x_wav, y_lab, ctype, sevty) in enumerate(self.test_loader):
                x_mel = x_wav.to(self.device)
                y_lab = y_lab.to(self.device)
                ctype = ctype.to(self.device)
                sevty = sevty.to(self.device)
                if flag:
                    print(x_wav.shape, y_lab.shape)
                    flag = False
                self.optimizer_Em.zero_grad()
                self.optimizer_vae.zero_grad()
                self.optimizer_cls.zero_grad()

                Loss_vae, Loss_attri, Loss_disen, Loss_cls = self.agedr.forward(x_input=x_mel, attri1=ctype, attri2=sevty, y_lab=y_lab)
                Loss_total = Loss_vae + Loss_attri + Loss_disen + Loss_cls
                Loss_total.backward()

                self.optimizer_cls.step()
                self.optimizer_vae.step()
                self.optimizer_Em.step()

            pbar.update(n=1)


if __name__ == '__main__':
    pass
