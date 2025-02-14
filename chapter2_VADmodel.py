# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2025-02-13 16:20
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import librosa
import torch
import torch.nn as nn
import torchaudio
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

WAVE_ROOT = "G:/DATAS-Medical/BILIBILICOUGH/"
NOISE_ROOT = "G:/DATAS-Medical/BILINOISE/"
name2label = {"breathe": 0, "cough": 2, "clearthroat": 1, "exhale": 3, "hum": 4, "inhale": 5, "noise": 6, "silence": 7,
              "sniff": 8, "speech": 9, "vomit": 10, "whooping": 11}


class TemporalAveragePooling(nn.Module):
    def __init__(self):
        """TAP
        Paper: Multi-Task Learning with High-Order Statistics for X-vector based Text-Independent Speaker Verification
        Link: https://arxiv.org/pdf/1903.12058.pdf
        """
        super(TemporalAveragePooling, self).__init__()

    def forward(self, x):
        """Computes Temporal Average Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels)
        """
        x = x.mean(dim=-1)
        # To be compatable with 2D input
        x = x.flatten(start_dim=1)
        return x


class TDNN_Extractor(nn.Module):
    def __init__(self, num_class=2, input_size=128, hidden_size=512, channels=1024, embd_dim=32):
        super(TDNN_Extractor, self).__init__()
        self.emb_size = embd_dim
        kernel_size, stride, padding = 1024, 488, 512
        self.wav2mel = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=kernel_size, stride=stride,
                                 padding=padding, bias=False)
        length = (22050 + 2 * padding - kernel_size) // stride + 1  # 87, who to 46?
        self.layer_norm = nn.LayerNorm(length)
        # self.wav2mel = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=1024, stride=512, padding=1024 // 2, bias=False)
        self.td_layer1 = torch.nn.Conv1d(in_channels=input_size, out_channels=hidden_size, dilation=1, kernel_size=5,
                                         stride=1)  # IW-5+1
        length = length - 5 + 1  # 83
        self.bn1 = nn.LayerNorm(length)
        self.td_layer2 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, dilation=2, kernel_size=3,
                                         stride=1, groups=hidden_size)  # IW-4+1
        length = length - 4  # 80
        self.bn2 = nn.LayerNorm(length)
        self.td_layer3 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, dilation=3, kernel_size=3,
                                         stride=1, groups=hidden_size)  # IW-6+1
        length = length - 6
        self.bn3 = nn.LayerNorm(length)
        self.td_layer4 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, dilation=1, kernel_size=1,
                                         stride=1, groups=hidden_size)  # IW+1
        length = length
        self.bn4 = nn.LayerNorm(length)
        self.td_layer5 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=channels, dilation=1, kernel_size=1,
                                         stride=1, groups=hidden_size)  # IW+1
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

        # self.pooling = TemporalAveragePooling()
        # self.bn5 = nn.BatchNorm1d(channels)
        # self.linear = nn.Linear(channels, embd_dim)
        # self.bn6 = nn.BatchNorm1d(embd_dim)
        # self.fc = nn.Linear(embd_dim, num_class)

    def forward(self, waveform):
        # x = x.transpose(2, 1)
        tgram = self.wav2mel(waveform)
        # print(tgram.shape)
        x = self.td_layer1(self.leakyrelu(self.layer_norm(tgram)))
        # print("shape of x as a wave:", x.shape)
        x = self.td_layer2(self.leakyrelu(self.bn1(x)))
        # print("shape of x in layer 1:", x.shape)
        x = self.td_layer3(self.leakyrelu(self.bn2(x)))
        # print("shape of x in layer 2:", x.shape)
        x = self.td_layer4(self.leakyrelu(self.bn3(x)))
        # print("shape of x in layer 3:", x.shape)
        out = self.td_layer5(self.leakyrelu(self.bn4(x)))
        # print("shape of x in layer 4:", out.shape)
        # out = self.bn5(self.pooling(x))
        # print("shape of x after pooling:", out.shape)
        # out = self.bn6(self.linear(out))
        # out = self.fc(out)
        return out


class VADModel(nn.Module):
    def __init__(self, n_mels=64, class_num=2):
        super().__init__()
        # Mel特征提取
        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, n_fft=2048, hop_length=512, n_mels=n_mels
        )

        # 波形分支（时域特征）
        # self.wave_conv = nn.Sequential(
        #     nn.Conv1d(1, 16, kernel_size=1024, stride=2, padding=2),
        #     nn.BatchNorm1d(16),
        #     nn.SiLU(),
        #     nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm1d(32),
        #     nn.SiLU(),
        # )
        self.wave_conv = TDNN_Extractor()

        # Mel分支（频域特征）
        self.mel_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((None, 32))  # 压缩频率维度
        )

        # # Transformer时序建模
        # encoder_layers = TransformerEncoderLayer(
        #     d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1
        # )
        # self.transformer = TransformerEncoder(encoder_layers, num_layers=4)

        # 分类头
        # self.reduction = nn.Sequential(
        #     nn.Conv1d(512, 32)
        # )
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.classifier = nn.Sequential(
            nn.Linear(512, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, class_num)
        )

    def forward(self, x):
        # x: (B, 1, 30000) 波形输入
        # 波形分支
        wave_feat = self.wave_conv(x)  # (B, 32, 7500)
        # print("wave_feat shape:", wave_feat.shape)
        wave_feat = wave_feat.permute(0, 2, 1)  # (B, 7500, 32)
        # print("wav feat shape:", wave_feat.shape)

        # 提取Mel特征
        mel = self.mel_extractor(x)  # .unsqueeze(1)  # (B, 1, n_mels, T)
        mel = torch.log(mel + 1e-6)  # 对数压缩
        # print("mel shape", mel.shape)

        # Mel分支
        mel_feat = self.mel_conv(mel)  # (B, 32, 64, 16)
        # print("mel feat shape:", mel_feat.shape)
        mel_feat = mel_feat.permute(0, 3, 1, 2).flatten(2)  # (B, 1024, 32)
        # print("mel feat shape:", mel_feat.shape)

        # 特征拼接
        combined = torch.cat([wave_feat, mel_feat], dim=-1)  # (B, 7500+1024, 32)
        # print("feat shape:", combined.shape)

        # # Transformer编码
        # src_key_padding_mask = (combined.mean(-1) == 0)  # 动态掩码
        # output = self.transformer(
        #     combined,  # .permute(1, 0, 2),
        #     src_key_padding_mask=src_key_padding_mask
        # )  # (T, B, d_model)
        # print(output.shape)
        # 分类
        output = self.pool(combined.mean(dim=1))
        # print(output.shape)
        logits = self.classifier(output)  # (B, 1)
        return logits  # .squeeze(-1)


def min2sec(t: str):
    parts = t.split(':')
    res = float(parts[-1])
    f = 60
    for i in range(len(parts) - 1):
        res += int(parts[len(parts) - 2 - i]) * f
        f *= 60
    return res


def get_bilicough_dataset():
    metadf = pd.read_csv(WAVE_ROOT + "bilicough_metainfo.csv", delimiter=',', header=0, index_col=None,
                         usecols=[0, 1, 2, 5], encoding="ansi")
    print(metadf)
    cur_fname = None
    cur_wav = None
    data_length = None
    sample_list = []
    label_list = []
    sr_list = []
    pre_st, pre_en = None, None
    # filename	st	en	labelfull	labelname	label	binlab
    for ind, item in enumerate(metadf.itertuples()):
        if (cur_fname != item[1]) or (cur_fname is None):
            cur_fname = item[1]
            cur_wav, sr = librosa.load(WAVE_ROOT + cur_fname + ".wav")
            if sr not in sr_list:
                sr_list.append(sr)
            data_length = sr
        st, en = int(min2sec(item[2]) * sr), int(min2sec(item[3]) * sr + 1)
        if en > len(cur_wav):
            en = len(cur_wav)
        if en - st < 100:
            raise Exception("Error Index.")
        sn = en - st
        # sec = (en - st)/22050
        if (pre_en is None):
            if st >= data_length:
                st_pos = 0
                ind = 0
                while st_pos + data_length <= st:
                    # if len(cur_wav[st_pos:st_pos+data_length]) != sr:
                    #     raise Exception("Error Length.")
                    sample_list.append(cur_wav[st_pos:st_pos + data_length])
                    label_list.append(0)
                    st_pos += data_length
                    ind += 1
                    if ind > 2:
                        break
                sample_list.append(cur_wav[st - data_length:st])
                label_list.append(0)
        else:
            if st - pre_en >= sr:
                st_pos = pre_en
                ind = 0
                while st_pos + data_length <= st:
                    # if len(cur_wav[st_pos:st_pos+data_length]) != sr:
                    #     raise Exception("Error Length.")
                    sample_list.append(cur_wav[st_pos:st_pos + data_length])
                    label_list.append(0)
                    st_pos += data_length
                    ind += 1
                    if ind > 2:
                        break
                sample_list.append(cur_wav[st - data_length:st])
                label_list.append(0)
        label = int(item[4])
        if sn == data_length:
            # if len(cur_wav[st:en]) != sr:
            #     raise Exception("Error Length.")
            sample_list.append(cur_wav[st:en])
            if label in [6, 7]:
                label_list.append(0)
            else:
                label_list.append(1)
        elif sn < data_length:
            new_sample = np.zeros(data_length)
            # print(st, en, sn, len(cur_wav), item[1])
            if en <= len(cur_wav):
                new_sample[:sn] = cur_wav[st:en]
            else:
                new_sample[:sn] = cur_wav[len(cur_wav) - sn:len(cur_wav)]
            # if len(new_sample) != sr:
            #     raise Exception("Error Length.")
            sample_list.append(new_sample)
            if label in [6, 7]:
                label_list.append(0)
            else:
                label_list.append(1)
        else:
            cnt_sum = sn // data_length + 1
            res = cnt_sum * data_length - sn
            overlap = res // (cnt_sum - 1)
            st_pos = st
            while st_pos + data_length < en:
                # if len(cur_wav[st_pos:st_pos+data_length]) < data_length:
                #     tmp_length = len(cur_wav[st_pos:st_pos+data_length])
                #     print(data_length, tmp_length)
                #     # raise Exception("Error Length.")
                #     print("Error Length.")
                #     new_sample = np.zeros(data_length)
                #     new_sample[:tmp_length] = cur_wav[st_pos:st_pos+data_length]
                #     sample_list.append(new_sample)
                # else:
                #     sample_list.append(cur_wav[st_pos:st_pos+data_length])
                sample_list.append(cur_wav[st_pos:st_pos + data_length])
                if label in [6, 7]:
                    label_list.append(0)
                else:
                    label_list.append(1)
                st_pos += data_length - overlap
            sample_list.append(cur_wav[en - data_length:en])
            label_list.append(1)
        pre_st, pre_en = st, en
    print("sound count:{}, all count:{}.".format(sum(label_list), len(label_list)))
    print("sample rate of waveform:", sr_list)
    return sample_list, label_list


# 添加噪声数据
def load_bilinoise_dataset(noise_length=22050, number=1):
    # noise_length = None
    filter_length = 25
    new_noise_list = []
    new_label_list = []
    flist = []
    for item in os.listdir(NOISE_ROOT):
        if item[-4:] == ".wav" and len(item) >= filter_length:
            flist.append(item)
    random.shuffle(flist)
    ind = 0
    for item in flist:
        cur_fname = NOISE_ROOT + item
        cur_wav, sr = librosa.load(cur_fname)
        noise_length = sr
        L = len(cur_wav)
        st_pos = np.random.randint(0, L - noise_length)
        new_noise_list.append(cur_wav[st_pos:st_pos + noise_length])
        new_label_list.append(0)
        # print(NOISE_ROOT+item)
        ind += 1
        if ind == number:
            break
    return new_noise_list, new_label_list


def addnoise(w, a):
    if len(w) != len(a):
        raise ValueError("The length of waveform and noise are not equal, and can't be added together!")
    return w + a


class BiliCoughDataset(Dataset):
    def __init__(self, audioseg, labellist, noises):
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


class Trainer2(object):
    def __init__(self):
        self.configs = {"batch_size": 32, "lr": 0.001, "epoch_num": 30}
        self.save_dir = "./runs/c2vadmodel/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.run_save_dir = self.save_dir + time.strftime("%Y%m%d%H%M", time.localtime()) + '/'
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.save_setting_str = "Model:{}, optimizer:{}, loss function:{}\n".format(
            "VADModel(wav TDNN + mel CNN + pool + mlp)", "Adam(lr={})".format(self.configs["lr"]),
            "nn.CrossEntropyLoss")
        self.save_setting_str += "dataset:{}, batch_size:{}, noise_p:{}\n".format("BiliCough+BiliNoise",
                                                                                self.configs["batch_size"], "0.333")
        self.save_setting_str += "epoch_num:{},\n".format(self.configs["epoch_num"])

    def __build_model(self):
        self.model = VADModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.configs["lr"])

    def __build_dataset(self):
        # step1:read data.
        sample_list, label_list = get_bilicough_dataset()
        # step2:detecting whether the data length is consistent, which indirectly detects the sampling rate and determines the fixed data length value.
        length_list = []
        for item in sample_list:
            if len(item) not in length_list:
                length_list.append(len(item))
        print("length of sample:", length_list)

        data_length = length_list[0]
        print("data length:", data_length)
        # step3:calculating the difference between the number of positive and negative samples.
        diff_num = 2 * sum(label_list) - len(label_list)
        print("need new negative sample of number {}".format(diff_num))
        if diff_num > 0:
            # If there are too many positive samples, read the noise data to construct negative samples and balance the positive and negative ratio.
            new_noise_list, new_label_list = load_bilinoise_dataset(noise_length=data_length, number=diff_num)
            sample_list.extend(new_noise_list)
            label_list.extend(new_label_list)
        elif diff_num < 0:
            # If the number of positive samples is less, do nothing.
            pass
        print("number of wave {} and label {}.".format(len(sample_list), len(label_list)))
        print("number of positive sample {} and label {}.".format(len(label_list), sum(label_list)))

        # 假设你已经有了train_loader和valid_loader
        noise_list, _ = load_bilinoise_dataset(noise_length=data_length, number=100)
        # step4: Scramble the data in the same order (using zip structure or setting the same random seeds can be completed).
        tmplist = list(zip(sample_list, label_list))
        random.shuffle(tmplist)
        sample_list, label_list = zip(*tmplist)
        trte_rate = int(len(sample_list) * 0.9)
        # then create training sets and verification sets with torch Library in a ratio of 9:1.
        # The batch_size is set to 32, in which the training set scrambles the data.
        self.train_loader = DataLoader(
            BiliCoughDataset(audioseg=sample_list[:trte_rate], labellist=label_list[:trte_rate], noises=noise_list),
            batch_size=self.configs["batch_size"], shuffle=True)
        self.valid_loader = DataLoader(
            BiliCoughDataset(audioseg=sample_list[trte_rate:], labellist=label_list[trte_rate:], noises=noise_list),
            batch_size=self.configs["batch_size"], shuffle=False)

    def train(self):
        self.__build_dataset()
        print("Start Training...")
        print("Build Model...")
        self.__build_model()
        print("Build Dataset...")
        flag = False
        Loss_Epoch_List = []
        for epoch_id in range(self.configs["epoch_num"]):
            self.model.train()
            Loss_Batch_List = []
            for batch_id, (x_wav, y_lab) in tqdm(enumerate(self.train_loader),
                                                 desc="Epoch:{} Training ".format(epoch_id)):
                self.optimizer.zero_grad()
                x_wav, y_lab = x_wav.to(self.device).unsqueeze(1).to(torch.float32), y_lab.to(self.device)  # .to(torch.float32)
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
                y_pred = self.model(x_wav)
                y_pred = np.argmax(y_pred.data.cpu().numpy(), axis=1)

                precision = metrics.precision_score(y_true=y_lab, y_pred=y_pred)
                recall = metrics.recall_score(y_true=y_lab, y_pred=y_pred)
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
        plt.savefig(self.run_save_dir+"vad_meanloss_epoch.png", dpi=300, format="png")
        plt.close(0)

        settingf = open(self.run_save_dir+"train_settings.txt", 'w')
        settingf.write(self.save_setting_str)
        settingf.write("loss:["+",".join([str(it) for it in Loss_Epoch_List])+']\n')
        settingf.write('precision:{}['.format(np.mean(pre_list))+",".join([str(it) for it in pre_list])+']\n')
        settingf.write('recall:{}['.format(np.mean(rec_list))+",".join([str(it) for it in rec_list])+']\n')
        settingf.write('accuracy:{}['.format(np.mean(acc_list))+",".join([str(it) for it in acc_list])+']\n')
        # plt.show()
        settingf.close()

        self.__save_model()

    def __valid(self):
        pass

    def test(self):
        pass

    def __save_model(self):
        torch.save(self.model.state_dict(),
                   self.run_save_dir+"vad_model_epoch{}.pth".format(self.configs["epoch_num"]))
        torch.save(self.optimizer.state_dict(),
                   self.run_save_dir+"vad_optimizer_epoch{}.pth".format(self.configs["epoch_num"]))
        print("models were saved.")

    def load_model(self):
        vad_model = VADModel()
        vad_model.load_state_dict(torch.load(self.save_dir+"202502141500/vad_model_epoch30.pth"))
        vad_model.eval()
        return vad_model

    def detection(self):
        vad_model = self.load_model()
        testwav, sr = librosa.load(WAVE_ROOT+"bilicough_009.wav")


if __name__ == '__main__':
    trainer = Trainer2()
    trainer.train()
    # print(tdnn(x).shape)
