#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/2/23 16:05
# @Author: ZhaoKe
# @File : chapter3_ADRmodel.py
# @Software: PyCharm
import os
import time
import itertools
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from models.conv_vae import ConvVAE
from modules.disentangle import AME
from modules.loss import vae_loss_fn, kl_2normal, pairwise_kl_loss
from readers.coughvid_reader import CoughVIDReader
from readers.noise_reader import load_bilinoise_dataset


class CoughDataset(Dataset):
    def __init__(self, audioseg, labellist, attri1_list, attri2_list, noises):
        self.audioseg = audioseg
        self.labellist = labellist
        self.attri1list = attri1_list
        self.attri2list = attri2_list
        self.noises = noises

    def __getitem__(self, ind):
        # When reading waveform data, add noise as data enhancement according to a 1/3 probability.
        if random.random() < 0.333:
            return self.audioseg[ind] + self.noises[random.randint(0, len(self.noises) - 1)], self.labellist[ind], \
                self.attri1list[ind], self.attri2list[ind]
        else:
            return self.audioseg[ind], self.labellist[ind], self.attri1list[ind], self.attri2list[ind]

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

        self.flag = True

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
        Loss_cls = self.cls_weight * self.categorical_loss(y_pred, y_lab)
        if self.flag:
            self.flag = False
            print("z_h:{}, a1.shape:{}, a2.sahpe:{}".format(z_latent.shape, mu1_latent.shape, mu2_latent.shape))
            print("pred:{}; x_Recon:{}".format(y_pred.shape, x_recon.shape))
            print("part[recon] recon loss:{};".format(Loss_recon))
        return x_recon, z_mu, Loss_vae, Loss_attri, Loss_disen, Loss_cls


class ADRTrainer(object):
    def __init__(self, mode="train"):
        self.mode = mode
        self.save_dir = "./runs/c3adrmodel/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.run_save_dir = self.save_dir + time.strftime("%Y%m%d%H%M", time.localtime()) + '/'

        # self.lambda_cat = 1.
        # self.lambda_con = 0.1
        self.img_size, self.channel = (64, 128), 1
        self.class_num = 2
        self.latent_dim = 32
        self.a1len, self.a2len = 12, 8
        self.blen = self.latent_dim - self.a1len - self.a2len
        self.configs = {
            "recon_weight": 0.01,
            "cls_weight": 1.,
            "kl_beta_alpha_weight": 0.01,
            "kl_c_weight": 0.075,
            "channels": 1,
            "class_num": 10,
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
        self.save_setting_str = ""
        self.save_setting_str += "dataset:{}, batch_size:{}, noise_p:{}\n".format(
            "CoughVID(length=32306, mel_shape=(dim=128, length=64))",
            self.configs["fit"]["batch_size"],
            "0.333")
        self.save_setting_str += "epoch_num:{},\n".format(self.configs["fit"]["epochs"])

    def __build_dataloader(self):
        cvr = CoughVIDReader(data_length=32306)
        print("waveform length:", cvr.data_length)
        sample_list, label_list, atri1, atri2 = cvr.get_sample_label_attris()
        tmplist = list(zip(sample_list, label_list, atri1, atri2))
        random.shuffle(tmplist)
        sample_list, label_list, atri1, atri2 = zip(*tmplist)
        trte_rate = int(0.9*len(sample_list)+1)
        noise_list, _ = load_bilinoise_dataset(NOISE_ROOT="G:/DATAS-Medical/BILINOISE/", noise_length=cvr.data_length,
                                               number=100)
        self.train_loader = DataLoader(
            CoughDataset(audioseg=sample_list[:trte_rate], labellist=label_list[:trte_rate],
                         noises=noise_list, attri1_list=atri1[:trte_rate],
                         attri2_list=atri2[trte_rate:]),
            batch_size=64, shuffle=True)
        self.train_loader = DataLoader(
            CoughDataset(audioseg=sample_list[trte_rate:], labellist=label_list[trte_rate:],
                         noises=noise_list, attri1_list=atri1[trte_rate:],
                         attri2_list=atri2[trte_rate:]),
            batch_size=64, shuffle=False)

    def __build_model(self):
        self.wav2mel = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=2048, hop_length=512,
                                                            n_mels=128).to(self.device)
        self.agedr = AGEDR(latent_dim=self.latent_dim, a1len=self.a1len, a2len=self.a2len, class_num=self.class_num).to(
            self.device)
        if self.mode == "train":
            self.optimizer_Em = torch.optim.Adam(
                itertools.chain(self.agedr.ame1.parameters(), self.agedr.ame2.parameters()), lr=0.0003,
                betas=(0.5, 0.999))
            self.optimizer_vae = torch.optim.Adam(self.agedr.vae.parameters(), lr=0.0001, betas=(0.5, 0.999))
            self.optimizer_cls = torch.optim.Adam(self.agedr.classifier.parameters(), lr=0.0002, betas=(0.5, 0.999))
            self.save_setting_str += "AGEDR(AME1(num=3->dim=12, lr=0.0003)+AME2(num=2->dim=8,lr=0.0003)+VAE(latent_dim=12,lr=0.0001)+Cls(class_num=2,lr=0.0002)),Adam.\n".format()
            self.save_setting_str += "align_weight={}, self.recon_weight={}, self.cls_weight={}, self.vae_weight={}, self.kl_attri_weight={}, self.kl_latent_weight={}\n".format(
                self.agedr.align_weight, self.agedr.recon_weight, self.agedr.cls_weight, self.agedr.vae_weight,
                self.agedr.kl_attri_weight,
                self.agedr.kl_latent_weight)

    def train(self):
        print("loading data.")
        self.__build_dataloader()
        print("data were loaded.\nbuilding model.")
        self.__build_model()
        print("models were built.")
        pbar = tqdm(total=self.configs["fit"]["epochs"])
        flag = True
        Loss_List_Epoch = []
        for epoch_id in range(self.configs["fit"]["epochs"]):
            Loss_List_Total = []
            Loss_List_disen = []
            Loss_List_attri = []
            Loss_List_vae = []
            Loss_List_cls = []
            x_mel = None
            x_recon = None
            pbar.set_description(desc="Epoch {}:".format(epoch_id))
            for batch_id, (x_wav, y_lab, ctype, sevty) in enumerate(self.train_loader):
                x_mel = self.wav2mel(x_wav.to(self.device)).unsqueeze(1)
                y_lab = y_lab.to(self.device)
                ctype = ctype.to(self.device)
                sevty = sevty.to(self.device)
                if flag:
                    print(x_wav.shape, y_lab.shape)
                    flag = False
                self.optimizer_Em.zero_grad()
                self.optimizer_vae.zero_grad()
                self.optimizer_cls.zero_grad()

                x_recon, z_mu, Loss_vae, Loss_attri, Loss_disen, Loss_cls = self.agedr(x_input=x_mel, attri1=ctype,
                                                                                       attri2=sevty, y_lab=y_lab)
                Loss_total = Loss_vae + Loss_attri + Loss_disen + Loss_cls
                Loss_total.backward()

                self.optimizer_cls.step()
                self.optimizer_vae.step()
                self.optimizer_Em.step()

                Loss_List_Total.append(Loss_total.item())
                Loss_List_disen.append(Loss_disen.item())
                Loss_List_attri.append(Loss_attri.item())
                Loss_List_vae.append(Loss_vae.item())
                Loss_List_cls.append(Loss_cls.item())

                if epoch_id == 0 and batch_id == 0:
                    print("part[cls] cls loss:{};".format(Loss_cls))
                    print("part[disen] beta alpha kl loss:{};".format(Loss_disen))
                    print("part[attri] pdf loss:{};".format(Loss_attri))
            Loss_List_Epoch.append([np.array(Loss_List_Total).mean(),
                                    np.array(Loss_List_disen).mean(),
                                    np.array(Loss_List_attri).mean(),
                                    np.array(Loss_List_vae).mean(),
                                    np.array(Loss_List_cls).mean()])
            # print("Loss Parts:")
            ns = ["total", "disen", "attri", "vae", "cls"]
            if epoch_id % 10 == 0:
                if epoch_id == 0:
                    if not os.path.exists(self.run_save_dir):
                        os.makedirs(self.run_save_dir, exist_ok=True)
                        with open(self.run_save_dir + "setting.txt", 'w') as fout:
                            fout.write(self.save_setting_str)
                else:
                    save_dir_epoch = self.run_save_dir + "epoch{}/".format(epoch_id)
                    os.makedirs(save_dir_epoch, exist_ok=True)
                    with open(save_dir_epoch + "losslist_epoch_{}.csv".format(epoch_id), 'w') as fin:
                        fin.write("total,disen,attri,vae,cls\n")
                        for epochlist in Loss_List_Epoch:
                            fin.write(",".join([str(it) for it in epochlist]) + '\n')
                    Loss_All_Lines = np.array(Loss_List_Epoch)
                    cs = ["black", "red", "green", "orange", "blue"]
                    # ns = ["total", "disen", "attri", "vae", "cls"]
                    for j in range(5):
                        plt.figure(j)
                        dat_lin = Loss_All_Lines[:, j]
                        plt.plot(range(len(dat_lin)), dat_lin, c=cs[j], alpha=0.7)
                        plt.title("Loss " + ns[j])
                        plt.savefig(save_dir_epoch + f'loss_{ns[j]}_iter_{epoch_id}.png')
                        plt.close(j)
                    torch.save(self.agedr.state_dict(), save_dir_epoch + "epoch_{}_agedr.pth".format(epoch_id))

                    torch.save(self.optimizer_Em.state_dict(),
                               save_dir_epoch + "epoch_{}_optimizer_Em.pth".format(epoch_id))
                    torch.save(self.optimizer_vae.state_dict(),
                               save_dir_epoch + "epoch_{}_optimizer_vae.pth".format(epoch_id))
                    torch.save(self.optimizer_cls.state_dict(),
                               save_dir_epoch + "epoch_{}_optimizer_cls.pth".format(epoch_id))

                    with open(save_dir_epoch + "ckpt_info_{}.txt".format(epoch_id), 'w') as fin:
                        fin.write("epoch:{}\n".format(epoch_id))
                        fin.write("total,disen,attri,vae,cls\n")
                        fin.write(",".join([str(it) for it in Loss_List_Epoch[-1]]) + '\n')
                    if x_recon is not None:
                        plt.figure(5)
                        img_to_origin = x_mel[:3].squeeze().data.cpu().numpy()
                        img_to_plot = x_recon[:3].squeeze().data.cpu().numpy()
                        for i in range(1, 4):
                            plt.subplot(3, 2, (i - 1) * 2 + 1)
                            plt.imshow(img_to_origin[i - 1])

                            plt.subplot(3, 2, (i - 1) * 2 + 2)
                            plt.imshow(img_to_plot[i - 1])
                        plt.savefig(save_dir_epoch + "recon_epoch-{}.png".format(epoch_id), format="png")
                        plt.close(5)
                    else:
                        raise Exception("x_recon is None.")
            pbar.update(n=1)

    def valid(self):
        self.__build_model()
        if self.mode == "valid":
            self.__load_checkpoint(self.save_dir + "202502251638/epoch30/epoch_30_agedr.pth")

    def __load_checkpoint(self, pretrain_path):
        self.agedr.load_state_dict(state_dict=torch.load(f=pretrain_path))


if __name__ == '__main__':
    trainer = ADRTrainer()
    trainer.train()
    # cvr = CoughVIDReader(data_length=32306)
    # print(cvr.data_length)
    # sample_list, label_list, atri1, atri2 = cvr.get_sample_label_attris()
    # noise_list, _ = load_bilinoise_dataset(NOISE_ROOT="G:/DATAS-Medical/BILINOISE/", noise_length=cvr.data_length,
    #                                        number=100)
    # train_loader = DataLoader(
    #     CoughDataset(audioseg=sample_list, labellist=label_list, noises=noise_list, attri1_list=atri1,
    #                  attri2_list=atri2),
    #     batch_size=64, shuffle=True)
    # for batch_id, (x_wav, y_lab, atr1, atr2) in tqdm(enumerate(train_loader),
    #                                                  desc="Training "):
    #     x_wav = x_wav.unsqueeze(1)
    #     print(x_wav.shape, y_lab.shape)
