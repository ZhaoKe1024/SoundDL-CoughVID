#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/11/13 20:57
# @Author: ZhaoKe
# @File : chapter2_SEDmodel.py
# @Software: PyCharm
import json
import os
import time
import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from readers.bilicough_reader import BiliCoughReader
from readers.neucough_reader import NEUCoughReader
from readers.coughvid_reader import CoughVIDReader
from readers.noise_reader import load_bilinoise_dataset
from models.tdnncnn import WSFNN


def get_combined_data():
    print("Build the Dataset consisting of BiliCough, NeuCough, CoughVID19.")
    bcr = BiliCoughReader()
    # ncr = NEUCoughReader()
    # cvr = CoughVIDReader()
    sample_list, label_list = [], []
    tmp_sl, tmp_ll = bcr.get_sample_label_list(mode="sed")
    sample_list.extend(tmp_sl)
    label_list.extend(tmp_ll)
    print("bilicough:", len(label_list), bcr.data_length)
    # tmp_sl, tmp_ll = ncr.get_sample_label_list(mode="cough")
    # sample_list.extend(tmp_sl)
    # label_list.extend(tmp_ll)
    # print("bilicough+neucough:", len(label_list), ncr.data_length)
    # tmp_sl, tmp_ll = cvr.get_sample_label_list()
    # sample_list.extend(tmp_sl)
    # label_list.extend(tmp_ll)
    # print("bilicough+neucough+coughvid:", len(label_list), cvr.data_length)
    # shuffle
    tmplist = list(zip(sample_list, label_list))
    random.shuffle(tmplist)
    sample_list, label_list = zip(*tmplist)

    noise_list, _ = load_bilinoise_dataset(NOISE_ROOT="G:/DATAS-Medical/BILINOISE/", noise_length=bcr.data_length,
                                           number=100)
    print("Loader noise data.")
    print("length of data:", len(sample_list), len(label_list), len(noise_list))
    print("data length:", bcr.data_length)
    return sample_list, label_list, noise_list


class CoughDataset(Dataset):
    def __init__(self, audioseg, labellist, noises=None):
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


class SEDModel(nn.Module):
    def __init__(self, class_num=6, latent_dim=1024):
        super().__init__()
        self.model = WSFNN(class_num=class_num, latent_dim=latent_dim)

    def forward(self, x, latent=False):
        return self.model(x=x, latent=latent)


def get_m2l(name: str):
    json_str = None  # json string
    with open("./configs/ucaslabel.json", 'r', encoding='utf_8') as fp:
        json_str = fp.read()
    json_data = json.loads(json_str)
    return json_data[name + "2label"], json_data["2label" + name]


class Trainer2SED(object):
    def __init__(self):
        self.configs = {"batch_size": 32, "lr": 0.001, "epoch_num": 30, "loss": "focal_loss"}
        self.save_dir = "./runs/c2sedmodel/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.run_save_dir = self.save_dir + time.strftime("%Y%m%d%H%M", time.localtime()) + '/'
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        json_str = None  # json string
        with open("./configs/ucaslabel.json", 'r', encoding='utf_8') as fp:
            json_str = fp.read()
        json_data = json.loads(json_str)
        self.sed_label2name = json_data["label2event"]
        self.save_setting_str = "Model:{}, optimizer:{}, loss function:{}\n".format(
            "SEDModel(wav TDNN + mel CNN + pool + mlp)", "Adam(lr={})".format(self.configs["lr"]),
            "nn.FocalLoss(class_num=6)")
        self.save_setting_str += "dataset:{}, batch_size:{}, noise_p:{}\n".format("BiliCough+BiliNoise",
                                                                                  self.configs["batch_size"], "0.333")
        self.save_setting_str += "epoch_num:{},\n".format(self.configs["epoch_num"])

    def __build_model(self):
        self.model = SEDModel(class_num=6).to(self.device)
        if self.configs["loss"] == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        elif self.configs["loss"] == "focal_loss":
            from modules.loss import FocalLoss
            self.criterion = FocalLoss(class_num=6)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.configs["lr"])

    def __build_dataset(self):
        sample_list, label_list, noise_list = get_combined_data()
        trte_rate = int(len(sample_list) * 0.9)

        self.train_loader = DataLoader(
            CoughDataset(audioseg=sample_list[:trte_rate], labellist=label_list[:trte_rate], noises=noise_list),
            batch_size=self.configs["batch_size"], shuffle=True)
        self.valid_loader = DataLoader(
            CoughDataset(audioseg=sample_list[trte_rate:], labellist=label_list[trte_rate:], noises=noise_list),
            batch_size=self.configs["batch_size"], shuffle=False)

    def train(self):
        self.__build_dataset()
        print("Build Model...")
        self.__build_model()
        print("Build Dataset...")
        flag = False
        Loss_Epoch_List = []
        print("Start Training...")
        for epoch_id in range(self.configs["epoch_num"]):
            self.model.train()
            Loss_Batch_List = []
            for batch_id, (x_wav, y_lab) in tqdm(enumerate(self.train_loader),
                                                 desc="Epoch:{} Training ".format(epoch_id)):
                self.optimizer.zero_grad()
                x_wav, y_lab = x_wav.to(self.device).unsqueeze(1).to(torch.float32), y_lab.to(
                    self.device)  # .to(torch.float32)
                if not flag:
                    print("shape of x y:", x_wav.shape, y_lab.shape)
                y_pred = self.model(x=x_wav)
                if not flag:
                    print("shape of pred:", y_pred.shape)
                # crossentropy(input=y_pred, target=t_lab)
                # focalloss(inputs=y_pred, targets=y_lab)
                loss_v = self.criterion(y_pred, y_lab)
                if not flag:
                    print("shape of loss_v:", loss_v.shape)
                    flag = True
                loss_v.backward()
                self.optimizer.step()
                Loss_Batch_List.append(loss_v.mean().item())
            Loss_Epoch_List.append(np.array(Loss_Batch_List).mean())
        self.model.eval()
        pre_list = []
        rec_list = []
        acc_list = []
        from sklearn import metrics
        for batch_id, (x_wav, y_lab) in tqdm(enumerate(self.valid_loader), desc="Testing..."):
            with torch.no_grad():
                x_wav = x_wav.to(self.device).unsqueeze(1).to(torch.float32)
                y_pred = self.model(x=x_wav)
                y_pred = np.argmax(y_pred.data.cpu().numpy(), axis=1)

                precision = metrics.precision_score(y_true=y_lab, y_pred=y_pred, average="micro")
                recall = metrics.recall_score(y_true=y_lab, y_pred=y_pred, average="micro")
                acc = metrics.accuracy_score(y_true=y_lab, y_pred=y_pred)
                pre_list.append(precision)
                rec_list.append(recall)
                acc_list.append(acc)
        print("precision:")
        print(pre_list)
        print("recall:")
        print(rec_list)
        print("accuracy:")
        print(acc_list)
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir, exist_ok=True)
        plt.figure(0)
        plt.plot(list(range(len(Loss_Epoch_List))), np.array(Loss_Epoch_List), c="black")
        plt.savefig(self.run_save_dir + "vad_meanloss_epoch.png", dpi=300, format="png")
        plt.close(0)

        settingf = open(self.run_save_dir + "train_settings.txt", 'w')
        settingf.write(self.save_setting_str)
        settingf.write("loss:[" + ",".join([str(it) for it in Loss_Epoch_List]) + ']\n')
        settingf.write('precision:{}['.format(np.mean(pre_list)) + ",".join([str(it) for it in pre_list]) + ']\n')
        settingf.write('recall:{}['.format(np.mean(rec_list)) + ",".join([str(it) for it in rec_list]) + ']\n')
        settingf.write('accuracy:{}['.format(np.mean(acc_list)) + ",".join([str(it) for it in acc_list]) + ']\n')
        # plt.show()
        settingf.close()

        self.__save_model()

    def __save_model(self):
        torch.save(self.model.state_dict(),
                   self.run_save_dir + "sed_model_epoch{}.pth".format(self.configs["epoch_num"]))
        torch.save(self.optimizer.state_dict(),
                   self.run_save_dir + "sed_optimizer_epoch{}.pth".format(self.configs["epoch_num"]))
        print("models were saved.")

    def load_model(self):
        vad_model = SEDModel()
        vad_model.load_state_dict(torch.load(self.save_dir + "202502141500/vad_model_epoch30.pth"))
        vad_model.eval()
        return vad_model


def min2sec(t: str):
    parts = t.split(':')
    res = float(parts[-1])
    f = 60
    for i in range(len(parts) - 1):
        res += int(parts[len(parts) - 2 - i]) * f
        f *= 60
    return res


def read_annotation(asspath: str):
    import librosa
    testwav, sr = librosa.load(asspath + ".wav")
    N = len(testwav)
    maxv = max(testwav)
    seg_list, label_list = [], []
    st, data_length, overlap = 0, int(sr), int(sr) // 2
    print("test waveform length:{}, sr:{}, overlap:{}.".format(N, sr, overlap))

    fin = open(asspath + ".ass", 'r', encoding="ANSI")
    line = fin.readline()
    while line[:8] != "[Events]":
        line = fin.readline()
    fin.readline()
    line = fin.readline()
    while line:
        parts = line.split(',')
        parts_tmp = None
        if '+' in parts[9]:
            parts_tmp = parts[9].strip().split('+')[0]
        else:
            parts_tmp = parts[9].strip()
        # print(parts_tmp)
        if '_' in parts[9]:
            parts_tmp = parts_tmp.split('_')[0]
            parts_tmp = parts_tmp.split('(')
        else:
            parts_tmp = parts_tmp.split('(')
        e_label, d_label = None, None
        if len(parts_tmp) == 2:
            e_label = parts_tmp[0]
            d_label = parts_tmp[1][:-1]
        elif len(parts_tmp) == 1:
            e_label = parts_tmp[0]
            d_label = "unknown"
        else:
            raise ValueError("Error at parts[9].split...")
        if e_label in ["music", "useless", "hum", "vomit", "sniff", "clearthroat", "wheeze"]:
            line = fin.readline()
            continue
        if e_label in ["breathe", "wheeze"]:
            raise ValueError("Error key:{}".format(e_label))

        st, en = min2sec(parts[1]), min2sec(parts[2])
        seg = testwav[int(st * sr):int(en * sr) + 1]
        # print("st:{}, en:{}, seg length:{}.".format(st, en, len(seg)))
        st = 0
        while st + data_length <= len(seg):
            seg_list.append(seg[st:st + data_length])
            label_list.append(e_label)
            st = st + data_length - overlap
        tmp = seg[-data_length:]
        if len(tmp) < data_length:
            tmp_wav = np.zeros(data_length)
            st = (data_length - len(tmp)) // 2
            tmp_wav[st:st + len(tmp)] = tmp
            seg_list.append(tmp_wav)
            label_list.append(e_label)
        line = fin.readline()

    # print(label_list)
    return seg_list, label_list


def detection(testwavname):
    from chapter2_VADmodel import VADModel
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    vad_model = VADModel()
    vad_model.load_state_dict(torch.load("./runs/c2vadmodel/202502141500/vad_model_epoch30.pth"))
    vad_model.to(device)
    vad_model.eval()
    print("load VAD model.")
    sed_model = SEDModel(class_num=6)
    sed_model.load_state_dict(torch.load("./runs/c2sedmodel/202503041630/sed_model_epoch30.pth"))
    sed_model.to(device)
    sed_model.eval()
    print("load SED model.")

    json_str = None  # json string
    with open("./configs/ucaslabel.json", 'r', encoding='utf_8') as fp:
        json_str = fp.read()
    json_data = json.loads(json_str)
    vad_label2name = json_data["label2vad"]
    sed_label2name = json_data["label2event"]
    # vad_name2label = json_data["event2label"]

    WAVE_ROOT = "G:/DATAS-Medical/BILIBILICOUGH/"
    # read_annotation(asspath=WAVE_ROOT + "bilicough_010")
    seg_list, label_list = read_annotation(asspath=WAVE_ROOT + testwavname)
    print("seg and label:", len(seg_list), len(label_list))
    seg_list = [torch.from_numpy(it) for it in seg_list]
    # ========================================================
    batch_size = 16
    x_batchs = []
    ind = 0
    while ind + batch_size < len(seg_list):
        x_batchs.append(torch.stack(seg_list[ind:ind + batch_size], dim=0))
        ind += batch_size
    x_batchs.append(torch.stack(seg_list[ind:], dim=0))
    print("batch num:{}, batch shape:{}".format(len(x_batchs), x_batchs[0].shape))
    print([len(it) for it in x_batchs])

    vad_pred_list = None
    for batch_id, x_wav in enumerate(x_batchs):
        with torch.no_grad():
            y_pred = vad_model(x=x_wav.to(device).unsqueeze(1).to(torch.float32))
            if vad_pred_list is None:
                vad_pred_list = y_pred
            else:
                vad_pred_list = torch.concat((vad_pred_list, y_pred), dim=0)
    vad_pred_list = np.argmax(vad_pred_list.data.cpu().numpy(), axis=1)
    print("VAD prediction:", vad_pred_list)
    vad_truth = [1] * len(label_list)
    for it in range(len(label_list)):
        if label_list[it] in ["silence", "noise"]:
            vad_truth[it] = 0

    print("truth and pred:", len(vad_truth), len(vad_pred_list))
    from sklearn import metrics
    precision = metrics.precision_score(y_true=vad_truth, y_pred=vad_pred_list)
    recall = metrics.recall_score(y_true=vad_truth, y_pred=vad_pred_list)
    acc = metrics.accuracy_score(y_true=vad_truth, y_pred=vad_pred_list)
    # print(precision(pred=vad_pred_list, target=vad_truth))
    print("precision:{}, recall:{}, accuracy:{}".format(precision, recall, acc))

    # =====================================================================================

    sound_events = []
    sound_indices = []
    event_labels = []
    m2l = json_data["event2label"]
    for i in range(len(vad_pred_list)):
        if vad_pred_list[i] > 0.5:
            sound_indices.append(i)
            sound_events.append(seg_list[i])
            event_labels.append(m2l[label_list[i]])

    batch_size = 16
    x_batchs = []
    sed_pred_list = []
    for x_wav in sound_events:
        with torch.no_grad():
            y_pred = sed_model(x=x_wav.to(device).unsqueeze(0).unsqueeze(1).to(torch.float32))
            sed_pred_list.append(np.argmax(y_pred.data.cpu().numpy()))
    print(sed_pred_list)
    precision = metrics.precision_score(y_true=event_labels, y_pred=sed_pred_list, average="micro")
    recall = metrics.recall_score(y_true=event_labels, y_pred=sed_pred_list, average="micro")
    acc = metrics.accuracy_score(y_true=event_labels, y_pred=sed_pred_list)
    # print(precision(pred=vad_pred_list, target=vad_truth))
    print("precision:{}, recall:{}, accuracy:{}".format(precision, recall, acc))

    """
    ---->bilicough_037:
    when (noise:0, silence:6)
    vad:precision:1.0, recall:0.9402985074626866, accuracy:0.9402985074626866
    sed:precision:0.7142857142857143, recall:0.7142857142857143, accuracy:0.7142857142857143
    when (noise, silence:0)
    precision:1.0, recall:0.9402985074626866, accuracy:0.9402985074626866
    precision:0.7142857142857143, recall:0.7142857142857143, accuracy:0.7142857142857143
    
    ---->bilicough_020:
    when (noise:0, silence:6)
    precision:1.0, recall:0.9392712550607287, accuracy:0.9418604651162791
    precision:0.7887931034482759, recall:0.7887931034482759, accuracy:0.7887931034482759
    when (noise silence:0)
    precision:0.8103448275862069, recall:0.9543147208121827, accuracy:0.7945736434108527
    precision:0.7887931034482759, recall:0.7887931034482759, accuracy:0.7887931034482759
    
    model 20250304
    precision:0.7974137931034483, recall:0.7974137931034483, accuracy:0.7974137931034483
    """


if __name__ == '__main__':
    # trainer = Trainer2SED()
    # trainer.train()

    # WAVE_ROOT = "G:/DATAS-Medical/BILIBILICOUGH/"
    # seg_list, ass_list = read_annotation(asspath=WAVE_ROOT + "bilicough_029")
    # for i in range(len(seg_list)):
    #     print(len(seg_list[i]), ass_list[i])

    # preprocessing_show()
    detection(testwavname="bilicough_020")
    # sed_label2name = {0: "breathe", 1: "clearthroat", 2: "cough", 3: "exhale", 4: "hum", 5: "inhale",
    #                   6: "sniff", 7: "speech", 8: "vomit", 9: "whooping"}
    # vad_name2labbel = {"breathe": 0, "clearthroat": 1, "cough": 2, "exhale": 3, "hum": 4, "inhale": 5,
    #                    "noise": 6, "silence": 7, "sniff": 8, "speech": 9, "vomit": 10, "whooping": 11}
    # sed_name2labbel = {"breathe": 0, "clearthroat": 1, "cough": 2, "exhale": 3, "hum": 4, "inhale": 5,
    #                    "sniff": 6, "speech": 7, "vomit": 8, "whooping": 9}
    # read_annotation("G:/DATAS-Medical/BILIBILICOUGH/bilicough_010.ass", vad_name2labbel)
