#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/11/13 20:57
# @Author: ZhaoKe
# @File : chapter2_SEDmodel.py
# @Software: PyCharm
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
    ncr = NEUCoughReader()
    cvr = CoughVIDReader()
    sample_list, label_list = [], []
    tmp_sl, tmp_ll = bcr.get_sample_label_list(mode="sed")
    sample_list.extend(tmp_sl)
    label_list.extend(tmp_ll)
    print("bilicough:", len(label_list), bcr.data_length)
    tmp_sl, tmp_ll = ncr.get_sample_label_list(mode="cough")
    sample_list.extend(tmp_sl)
    label_list.extend(tmp_ll)
    print("bilicough+neucough:", len(label_list), ncr.data_length)
    tmp_sl, tmp_ll = cvr.get_sample_label_list()
    sample_list.extend(tmp_sl)
    label_list.extend(tmp_ll)
    print("bilicough+neucough+coughvid:", len(label_list), cvr.data_length)
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
    def __init__(self, class_num=10):
        super().__init__()
        self.model = WSFNN(class_num=class_num)

    def forward(self, x):
        return self.model(x=x)


class Trainer2SED(object):
    def __init__(self):
        self.configs = {"batch_size": 32, "lr": 0.001, "epoch_num": 30}
        self.save_dir = "./runs/c2sedmodel/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.run_save_dir = self.save_dir + time.strftime("%Y%m%d%H%M", time.localtime()) + '/'
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sed_label2name = {0: "breathe", 1: "clearthroat", 2: "cough", 3: "exhale", 4: "hum", 5: "inhale",
                               6: "sniff", 7: "speech", 8: "vomit", 9: "whooping"}
        self.save_setting_str = "Model:{}, optimizer:{}, loss function:{}\n".format(
            "SEDModel(wav TDNN + mel CNN + pool + mlp)", "Adam(lr={})".format(self.configs["lr"]),
            "nn.CrossEntropyLoss")
        self.save_setting_str += "dataset:{}, batch_size:{}, noise_p:{}\n".format("BiliCough+BiliNoise",
                                                                                  self.configs["batch_size"], "0.333")
        self.save_setting_str += "epoch_num:{},\n".format(self.configs["epoch_num"])

    def __build_model(self):
        self.model = SEDModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
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
                loss_v = self.criterion(input=y_pred, target=y_lab)
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


def read_annotation(asspath: str, sedm2l: dict):
    fin = open(asspath, 'r', encoding="ANSI")
    line = fin.readline()
    label_list = []
    while line[:8] != "Dialogue":
        line = fin.readline()
    while line:
        # print(line)
        parts = line.split(',')
        lab_tmp = parts[9].strip()
        if lab_tmp == "useless":
            pass
        else:
            label = None
            # label_list.append(lab_tmp)
            if lab_tmp[:3] == "hum":
                label = lab_tmp[:3]
            elif lab_tmp[:5] in ["cough", "noise", "sniff", "vomit"]:
                label = lab_tmp[:5]
            elif lab_tmp[:6] in ["inhale", "exhale", "speech"]:
                label = lab_tmp[:6]
            elif lab_tmp[:7] in ["breathe", "silence"]:
                label = lab_tmp[:7]
            elif lab_tmp[:8] in ["whooping"]:
                label = lab_tmp[:8]
            elif lab_tmp[:11] in ["clearthroat"]:
                label = lab_tmp[:11]
            else:
                print(lab_tmp)
                raise Exception("Unknown Class.")
            label_list.append(sedm2l[label])
        line = fin.readline()

    # print(label_list)
    return label_list


def detection():
    import librosa
    from chapter2_VADmodel import VADModel
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    vad_model = VADModel()
    vad_model.load_state_dict(torch.load("./runs/c2vadmodel/202502141500/vad_model_epoch30.pth"))
    vad_model.to(device)
    vad_model.eval()
    print("load VAD model.")
    sed_model = SEDModel()
    sed_model.load_state_dict(torch.load("./runs/c2sedmodel/202502161815/sed_model_epoch30.pth"))
    sed_model.to(device)
    sed_model.eval()
    print("load SED model.")

    WAVE_ROOT = "G:/DATAS-Medical/BILIBILICOUGH/"
    testwav, sr = librosa.load(WAVE_ROOT + "bilicough_009.wav")
    N = len(testwav)
    maxv = max(testwav)
    seg_list = []
    st, step, overlap = 0, 22050, 22050 // 3
    while st + step <= N:
        seg_list.append(testwav[st:st + step])
        st = st + step - overlap
    tmp = testwav[st:]
    new_tmp = np.zeros(step)
    st = (step - len(tmp)) // 2
    new_tmp[st:st + len(tmp)] = tmp
    seg_list.append(new_tmp)
    print(len(seg_list))

    seg_list = [torch.from_numpy(it) for it in seg_list]

    batch_size = 32
    x_batchs = []
    ind = 0
    while ind + batch_size < len(seg_list):
        x_batchs.append(torch.stack(seg_list[ind:ind + batch_size], dim=0))
        ind += batch_size
    x_batchs.append(torch.stack(seg_list[ind:], dim=0))
    print("batch num:{}, batch shape:{}".format(len(x_batchs), x_batchs[0].shape))

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

    sound_events = []
    indices = []
    for i in range(len(vad_pred_list)):
        if vad_pred_list[i] > 0.5:
            indices.append(i)
            sound_events.append(seg_list[i])
    print("sound event indicex:", indices)
    # print(sound_events)

    batch_size = 32
    x_batchs = []
    ind = 0
    while ind + batch_size < len(sound_events):
        x_batchs.append(torch.stack(sound_events[ind:ind + batch_size], dim=0))
        ind += batch_size
    x_batchs.append(torch.stack(sound_events[ind:], dim=0))
    print("batch num:{}, batch shape:{}".format(len(x_batchs), x_batchs[0].shape))
    pred_list = None
    for batch_id, x_wav in enumerate(x_batchs):
        with torch.no_grad():
            y_pred = sed_model(x=x_wav.to(device).unsqueeze(1).to(torch.float32))
            if pred_list is None:
                pred_list = y_pred
            else:
                pred_list = torch.concat((pred_list, y_pred), dim=0)
    pred_list = np.argmax(pred_list.data.cpu().numpy(), axis=1)
    print(pred_list)

    sed_label2name = {0: "breathe", 1: "clearthroat", 2: "cough", 3: "exhale", 4: "hum", 5: "inhale",
                      6: "sniff", 7: "speech", 8: "vomit", 9: "whooping", -1: "--"}
    vad_name2labbel = {"breathe": 0, "clearthroat": 1, "cough": 2, "exhale": 3, "hum": 4, "inhale": 5,
                       "noise": 6, "silence": 7, "sniff": 8, "speech": 9, "vomit": 10, "whooping": 11}

    vad_label2name = {0: "breathe", 1: "clearthroat", 2: "cough", 3: "exhale", 4: "hum", 5: "inhale",
                      6: "noise", 7: "silence", 8: "sniff", 9: "speech", 10: "vomit", 11: "whooping"}
    result = [7] * len(vad_pred_list)
    for ind in range(len(indices)):
        if pred_list[ind] > 5:
            result[indices[ind]] = pred_list[ind] + 2
        else:
            result[indices[ind]] = pred_list[ind]
    # print([vad_label2name[it] for it in result])

    result_squeezed = []
    it = result[0]
    ind = 1
    while ind < len(result):
        if result[ind] != it:
            result_squeezed.append(it)
            it = result[ind]
        ind += 1
    print(result_squeezed)
    print([vad_label2name[it] for it in result_squeezed])

    print("groundtruth:")
    print(read_annotation(asspath=WAVE_ROOT + "bilicough_010.ass", sedm2l=vad_name2labbel))


if __name__ == '__main__':
    trainer = Trainer2SED()
    trainer.train()
    # preprocessing_show()
    # detection()
    # sed_label2name = {0: "breathe", 1: "clearthroat", 2: "cough", 3: "exhale", 4: "hum", 5: "inhale",
    #                   6: "sniff", 7: "speech", 8: "vomit", 9: "whooping"}
    # vad_name2labbel = {"breathe": 0, "clearthroat": 1, "cough": 2, "exhale": 3, "hum": 4, "inhale": 5,
    #                    "noise": 6, "silence": 7, "sniff": 8, "speech": 9, "vomit": 10, "whooping": 11}
    # sed_name2labbel = {"breathe": 0, "clearthroat": 1, "cough": 2, "exhale": 3, "hum": 4, "inhale": 5,
    #                    "sniff": 6, "speech": 7, "vomit": 8, "whooping": 9}
    # read_annotation("G:/DATAS-Medical/BILIBILICOUGH/bilicough_010.ass", vad_name2labbel)
