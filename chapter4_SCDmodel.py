#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/2/17 21:46
# @Author: ZhaoKe
# @File : chapter4_SCDmodel.py
# @Software: PyCharm

# Sound Causality Diagnosis

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class SCDModel(nn.Module):
    def __init__(self, class_num=10, multi_event_length=8, fuse_model="attn", cls_model="mlp", latent_dim=1024, hidden_dim=64):
        super().__init__()
        self.vad_model = WSFNN(class_num=2)
        self.extractor = WSFNN(class_num=class_num, latent_dim=latent_dim)
        self.multi_event_length = multi_event_length
        self.event_dim = (latent_dim//2) // self.multi_event_length
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
        adj = torch.ones(self.multi_event_length, self.multi_event_length)
        # 对角线也可设为1(含自环)，或根据需要设为0
        # adj = adj.fill_diagonal_(0)  # 如果不想要自环，可以这样做
        self.register_buffer('adj', adj)

        print("Pooling after Fusioning the TDNN and CNN.")
        self.pool = nn.MaxPool1d(kernel_size=4)
        # 最终分类器
        self.classifier = nn.Linear(hidden_dim, class_num)
        # 构造一个全连接层以便在两种模式下可以统一输出
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # # pred_activity_silence, latent_vector(representation)
        # pred_as, latent_as = self.vad_model(x, latent=True)

        # pred_sound_event, latent_vector(representation)
        pred_se, latent_v = self.extractor(x, latent=True)  # _, (bs, 32, 2048)
        latent_v = self.pool(latent_v.mean(dim=1))  # torch.Size([32, 512])
        print("latent v:", latent_v.shape)
        bs, v_dim = latent_v.shape
        # latent vector of multi-event vector
        latent_me = latent_v.view(bs, self.multi_event_length, self.event_dim)
        print("latent_me:", latent_me.shape)
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
        print("out shape:", out.shape)
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits


class Trainer4SCD(object):
    def __init__(self):
        self.sed_label2name = {0: "breathe", 1: "clearthroat", 2: "cough", 3: "exhale", 4: "hum", 5: "inhale",
                               6: "sniff", 7: "speech", 8: "vomit", 9: "whooping"}
        self.configs = {"batch_size": len(self.sed_label2name) * (64 // len(self.sed_label2name)),
                        "lr": 0.001, "epoch_num": 30}
        self.save_dir = "./runs/c2scdmodel/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.run_save_dir = self.save_dir + time.strftime("%Y%m%d%H%M", time.localtime()) + '/'
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == '__main__':
    Trainer4SCD()
    # x = torch.rand(size=(64, 1, 22050))
    # m1 = SCDModel(fuse_model="attn")
    # print(m1(x).shape)
    # print("=======================")
    # m2 = SCDModel(fuse_model="gnn")
    # print(m2(x).shape)
