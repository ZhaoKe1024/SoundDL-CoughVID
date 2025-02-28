#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/2/17 21:46
# @Author: ZhaoKe
# @File : chapter4_SCDmodel.py
# @Software: PyCharm

# Sound Causality Diagnosis

import os
import time
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models.tdnncnn import WSFNN
from modules.attentions import SimpleSelfAttention, SimpleGCNLayer


def get_combined_batchs():
    print("Build the Dataset consisting of BiliCough, NeuCough, CoughVID19.")
    # bcr = BiliCoughReader()
    # ncr = NEUCoughReader()
    # cvr = CoughVIDReader()
    # bcr.get_multi_event_batches()
    # ncr.get_multi_event_batches()
    # cvr.get_multi_event_batches()


def simulate_data():
    batch_num = 10
    batch_size = 32
    seg_num = 9
    sample_len = 22050
    sample_list = [np.random.rand(sample_len) for _ in range(batch_num * batch_size * seg_num)]
    label_list = [np.random.randint(0, 5) for _ in range(batch_num * batch_size * seg_num)]
    return sample_list, label_list


class SCDCoughDataset(Dataset):
    def __init__(self, audioseg, labellist, noises, mode="valid"):
        self.audioseg = audioseg
        self.labellist = labellist
        self.noises = noises
        self.mode = mode

    def __getitem__(self, ind):
        # When reading waveform data, add noise as data enhancement according to a 1/3 probability.
        if (self.mode == "train") and (random.random() < 0.333):
            return self.audioseg[ind] + self.noises[random.randint(0, len(self.noises) - 1)], \
                self.labellist[ind]
        else:
            return self.audioseg[ind], self.labellist[ind]

    def __len__(self):
        return len(self.audioseg)


class SCDModel(nn.Module):
    def __init__(self, class_num=10, seg_num=5, fuse_model="attn", cls_model="mlp", latent_dim=1024, hidden_dim=64):
        super().__init__()
        print("Build SCDModel, seg_num:{}, class_num:{}.".format(seg_num, class_num))
        self.vad_model = WSFNN(class_num=2)
        self.extractor = WSFNN(class_num=class_num, latent_dim=latent_dim)

        self.seg_num = seg_num
        self.event_dim = 512
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
        adj = torch.ones(self.seg_num, self.seg_num)
        # 对角线也可设为1(含自环)，或根据需要设为0
        # adj = adj.fill_diagonal_(0)  # 如果不想要自环，可以这样做
        self.register_buffer('adj', adj)

        print("Pooling after Fusioning the TDNN and CNN.")
        self.pool = nn.MaxPool1d(kernel_size=4)
        # 最终分类器
        self.classifier = nn.Linear(hidden_dim, class_num)
        # 构造一个全连接层以便在两种模式下可以统一输出
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x_input):
        # # pred_activity_silence, latent_vector(representation)
        # pred_as, latent_as = self.vad_model(x, latent=True)

        # pred_sound_event, latent_vector(representation)
        pred_se, latent_v = self.extractor(x_input, latent=True)  # _, (bs, 32, 2048)
        latent_v = self.pool(latent_v.mean(dim=1))  # torch.Size([bs, 512])
        # print("latent v:", latent_v.shape)

        bs, v_dim = latent_v.shape
        # latent vector of multi-event vector
        latent_me = latent_v.view(-1, self.seg_num, v_dim)
        # print("latent_me:", latent_me.shape)
        if self.fuse_model == "attn":
            out, attn_weights = self.fuse_layer(latent_me)
            out = self.attention_fc(out)
            out = F.relu(out)
            out = out.mean(dim=1)
        elif self.fuse_model == "gnn":
            out = self.fuse_layer(latent_me, self.adj)
            out = F.relu(out)
            out = out.mean(dim=1)
        else:
            raise ValueError("Unknown fused model:{}(at forward().).".format(self.fuse_model))
        # print("out shape:", out.shape)
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits


class Trainer4SCD(object):
    def __init__(self, mode="train"):
        self.mode = mode
        self.sed_label2name = {0: "breathe", 1: "clearthroat", 2: "cough", 3: "exhale", 4: "hum", 5: "inhale",
                               6: "sniff", 7: "speech", 8: "vomit", 9: "whooping"}
        # class_num: cold, asthma, pneumonia, whooping, bronchitis(COPD, bronchiolitis)
        self.configs = {"batch_size": len(self.sed_label2name) * (64 // len(self.sed_label2name)),
                        "lr": 0.001, "epoch_num": 30,
                        "data": {"series_second": 5, "sr": 22050, "overlap": 11025, "seg_num": 9},
                        "model": {"class_num": 5}
                        }
        self.save_dir = "./runs/c2scdmodel/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.run_save_dir = self.save_dir + time.strftime("%Y%m%d%H%M", time.localtime()) + '/'
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __build_dataloader(self):
        self.train_loader = None
        test_wav, test_label = simulate_data()
        self.valid_loader = DataLoader(SCDCoughDataset(audioseg=test_wav, labellist=test_label, noises=[], mode="valid"), batch_size=32*self.configs["data"]["seg_num"])

    def __build_models(self):
        self.model = SCDModel(seg_num=self.configs["data"]["seg_num"], class_num=self.configs["model"]["class_num"],
                              fuse_model="attn").to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        if self.mode == "train":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.configs["lr"])

    def train(self):
        print("loading data.")
        self.__build_dataloader()
        print("data were loaded.\nbuilding model.")
        self.__build_models()
        print("models were built.")
        pbar = tqdm(total=self.configs["epoch_num"])
        for epoch_id in range(self.configs["epoch_num"]):
            pbar.set_description(desc="Epoch {}:".format(epoch_id))
            for batch_id, (x_wav, y_lab) in enumerate(self.train_loader):
                x_wav = x_wav.to(self.device).float()
                y_lab = y_lab.to(self.device).long()
                self.optimizer.zero_grad()
                y_pred = self.model(x_input=x_wav)
                loss_v = self.criterion(input=y_pred, target=y_lab)
                loss_v.backward()
                self.optimizer.step()

    def valid(self):
        print("loading data.")
        self.__build_dataloader()
        print("data were loaded.\nbuilding model.")
        self.__build_models()
        print("models were built.")
        pbar = tqdm(total=self.configs["epoch_num"])
        flag = True
        for epoch_id in range(self.configs["epoch_num"]):
            pbar.set_description(desc="Valid Epoch {}:".format(epoch_id))
            for batch_id, (x_wav, y_lab) in enumerate(self.valid_loader):
                x_wav = x_wav.to(self.device).unsqueeze(1).float()
                y_lab = y_lab.to(self.device).long()
                if flag:
                    print(x_wav.shape, y_lab.shape)
                # self.optimizer.zero_grad()
                y_pred = self.model(x_input=x_wav)
                if flag:
                    print(y_pred.shape)
                    flag = False
                # loss_v = self.criterion(input=y_pred, target=y_lab)
                # loss_v.backward()
                # self.optimizer.step()
            pbar.update(1)


if __name__ == '__main__':
    trainer = Trainer4SCD(mode="valid")
    trainer.valid()
    # x = torch.rand(size=(64, 1, 22050))
    # m1 = SCDModel(fuse_model="attn")
    # print(m1(x).shape)
    # print("=======================")
    # m2 = SCDModel(fuse_model="gnn")
    # print(m2(x).shape)
