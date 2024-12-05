#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/12/5 11:03
# @Author: ZhaoKe
# @File : SilenceDetection.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class NEUCOUGHDataset(Dataset):
    def __init__(self, segs_list, label_list):
        self.segs_list = segs_list
        self.label_list = label_list

    def __getitem__(self, ind):
        return self.segs_list[ind], self.label_list[ind]

    def __len__(self):
        return len(self.label_list)


class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512, num_class=2):
        super(TgramNet, self).__init__()
        # if "center=True" of stft, padding = win_len / 2
        self.conv_extrctor = nn.Conv1d(1, mel_bins, kernel_size=win_len, stride=hop_len, padding=win_len // 2,
                                       bias=False)

        self.conv_encoder = nn.Sequential()
        self.conv_encoder.append(nn.LayerNorm(20))
        self.conv_encoder.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv_encoder.append(nn.Conv1d(mel_bins, mel_bins // 2, kernel_size=3, stride=1, padding=1, bias=False))

        self.pooling = nn.Sequential()
        self.pooling.append(nn.MaxPool2d(4))
        self.pooling.append(nn.Flatten(start_dim=1))

        self.classifier = nn.Linear((mel_bins // 2 * 20) // (4 * 4), num_class)

    def forward(self, x):
        out = self.conv_extrctor(x)
        # print("conv1d:", out.shape)
        out = self.conv_encoder(out)
        # print("encoder:", out.shape)
        out = self.pooling(out)
        # print("pooling:", out.shape)
        return self.classifier(out)


def min2sec(t: str):
    parts = t.split(':')
    return int(parts[1]) * 60 + float(parts[2])


def read_ass_to_segs(asspath):
    fin = open(asspath, 'r', encoding="UTF-8")
    offset = 20
    while offset > 0:
        fin.readline()
        offset -= 1
    intervals = []
    line = fin.readline()
    while line:
        parts = line.strip().split(',')
        # print(parts[1], parts[2], parts[-1])
        intervals.append((min2sec(parts[1]), min2sec(parts[2])))
        line = fin.readline()
    fin.close()
    return intervals


def read_audio_for_silence(wavname):
    intervals = read_ass_to_segs("F:/DATAS/NEUCOUGHDATA_FULL/{}_annotations.ass".format(wavname))
    print(intervals)
    sample, sr = librosa.load("F:/DATAS/NEUCOUGHDATA_FULL/{}_audiodata_元音字母a.wav".format(wavname))
    step, overlap = 8000, 2000
    st_pos, en_pos = 0, step
    st_pre, en_pre = -6001, -1
    Length = len(sample)
    st_cur, en_cur = int(intervals[0][0] * sr), int(intervals[0][1] * sr)
    st_tail, en_tail = int(intervals[1][0] * sr), int(intervals[1][1] * sr)
    jdx = 2
    Segs_List = []
    label_List = []
    tr = 0.3 * (step + overlap)
    while en_cur < Length:
        if st_pos > st_cur:
            if en_pos < en_cur:
                label_List.append(1)  # ()([])()
            elif st_pos < en_cur:
                if en_cur - st_pos > tr:
                    label_List.append(1)  # ()([)]()
                else:
                    label_List.append(0)
            elif st_pos > en_cur:
                if en_pos > st_tail:
                    if en_pos - st_tail > tr:
                        label_List.append(1)  # ()()[(])
                    else:
                        label_List.append(0)
                else:
                    label_List.append(0)  # ()()[]()
        elif en_pos > st_cur:
            if en_pos - st_cur > tr:
                label_List.append(1)  # ()[(])()
            else:
                label_List.append(0)
        elif st_pos < en_pre:
            if en_pre - st_pos > tr:
                label_List.append(1)  # ([)]()()
            else:
                label_List.append(0)
        else:
            label_List.append(0)  # ()[]()()
        # print("st_pos:{}, en_pos:{},\nst_cur:{}, en_cur:{},\njdx:{}, interval:{},\nlen(label_List):{}, Length / step:{}, Length:{}".format(st_pos, en_pos, st_cur, en_cur, jdx, len(intervals), len(label_List), Length / step, Length))
        Segs_List.append(sample[st_pos:en_pos + overlap])
        if st_pos > en_cur:
            st_pre, en_pre = st_cur, en_cur
            st_cur, en_cur = st_tail, en_tail
            if jdx < len(intervals):
                st_tail, en_tail = int(intervals[jdx][0] * sr), int(intervals[jdx][1] * sr)
                jdx += 1
            else:
                st_tail, en_tail = Length + 1, Length + step
        st_pos, en_pos = en_pos, en_pos + step
    return Segs_List, label_List


if __name__ == '__main__':
    ROOT = "F:/DATAS/NEUCOUGHDATA_COUGHSINGLE/"
    slice_raw = pd.read_csv(ROOT + 'neucough_metainfo_slice.txt', header=0, index_col=None)
    Segs_List, label_List = read_audio_for_silence(wavname="20240921111118")
    print(len(Segs_List), '\n', label_List)

    split_pos = int(len(label_List) * 0.9)
    train_data, train_label = torch.from_numpy(np.array(Segs_List[:split_pos])), torch.from_numpy(
        np.array(label_List[:split_pos]))
    valid_data, valid_label = torch.from_numpy(np.array(Segs_List[split_pos:])), torch.from_numpy(
        np.array(label_List[split_pos:]))
    print("train_data:{}, train_label:{}, valid_data:{}, valid_label:{}".format(train_data.shape, train_label.shape,
                                                                                valid_data.shape, valid_label.shape))

    train_dataset = NEUCOUGHDataset(train_data, train_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True,
                              pin_memory=True,
                              num_workers=0)
    test_dataset = NEUCOUGHDataset(valid_data, valid_label)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False,
                             pin_memory=True,
                             num_workers=0)
    print("Data Split into:", len(train_dataset), len(test_dataset))

    tgram = TgramNet(num_layer=2, num_class=2)
    print(tgram)
    input_wav = torch.rand(32, 1, 10000)
    print(tgram(input_wav).shape)

    device = torch.device("cuda")
    model = TgramNet(num_layer=2, num_class=2).to(device)
    print(model)
    loss_f1 = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004, betas=(0.5, 0.999))
    Loss_List = []
    for epoch_id in tqdm(range(20), desc=">>Epoch:"):
        # print()
        model.train()
        for idx, (x_input, y_label) in enumerate(train_loader):
            x_input = x_input.unsqueeze(1).to(device)
            y_label = y_label.to(device).to(torch.long)
            optimizer.zero_grad()
            y_pred = model(x_input)
            # print(y_pred.shape, y_label.shape)
            loss_v = loss_f1(input=y_pred, target=y_label)
            Loss_List.append(loss_v.item())
            loss_v.backward()
            optimizer.step()
        # print(Loss_List)

        model.eval()
        y_preds = None
        y_labels = None
        for idx, (x_input, y_label) in tqdm(enumerate(train_loader), desc=">>Valid Train:"):
            x_input = x_input.unsqueeze(1).to(device)
            y_label = y_label.to(device)
            if y_labels is None:
                y_labels = y_label
            else:
                y_labels = torch.concat((y_labels, y_label), dim=0)

            with torch.no_grad():
                y_pred = model(x_input)
            if y_preds is None:
                y_preds = y_pred
            else:
                y_preds = torch.concat((y_preds, y_pred), dim=0)
        from sklearn import metrics

        y_labs = y_labels.data.cpu().numpy()
        y_preds = y_preds.data.cpu().numpy()
        y_preds_label = y_preds.argmax(-1)
        precision = metrics.precision_score(y_labs, y_preds_label)
        recall = metrics.recall_score(y_labs, y_preds_label)
        acc = metrics.accuracy_score(y_labs, y_preds_label)
        print("\nTrain Dataset Result:", precision, recall, acc)

        y_preds = None
        y_labels = None
        for idx, (x_input, y_label) in tqdm(enumerate(test_loader), desc=">>Valid Test:"):
            x_input = x_input.unsqueeze(1).to(device)
            y_label = y_label.to(device)
            if y_labels is None:
                y_labels = y_label
            else:
                y_labels = torch.concat((y_labels, y_label), dim=0)

            with torch.no_grad():
                y_pred = model(x_input)
            if y_preds is None:
                y_preds = y_pred
            else:
                y_preds = torch.concat((y_preds, y_pred), dim=0)
        y_labs = y_labels.data.cpu().numpy()
        y_preds = y_preds.data.cpu().numpy()
        y_preds_label = y_preds.argmax(-1)
        precision = metrics.precision_score(y_labs, y_preds_label)
        recall = metrics.recall_score(y_labs, y_preds_label)
        acc = metrics.accuracy_score(y_labs, y_preds_label)
        print("Test Dataset Result:", precision, recall, acc)
        # if precision > 0.96 and recall > 0.96:
        #     save_dir = "./runs/sed_crnn/" + time.strftime("%Y%m%d%H%M", time.localtime()) + "/"
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir, exist_ok=True)
        #     torch.save(model.state_dict(), save_dir + "epoch_{}_sedmodel.pth".format(epoch_id))
        #     torch.save(optimizer.state_dict(), save_dir + "epoch_{}_optimizer.pth".format(epoch_id))
