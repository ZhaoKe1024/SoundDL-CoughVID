#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/12/4 15:54
# @Author: ZhaoKe
# @File : test_code.py
# @Software: PyCharm
"""
暂时测试代码用
"""
import random

import numpy as np
from tqdm import tqdm
import torch
from readers.bilicough_reader import BiliCoughReader
from readers.neucough_reader import NEUCoughReader
from readers.coughvid_reader import CoughVIDReader
from readers.noise_reader import load_bilinoise_dataset
from torch.utils.data import Dataset, DataLoader


def redundancy_overlap_generate():
    length = 46
    data_length = 11
    cnt_sum = length // data_length + 1
    res = cnt_sum * data_length - length
    print(cnt_sum, res)
    overlap = res // (cnt_sum - 1)
    print(overlap)
    st = 0
    while st + data_length <= length:
        print("[{}, {}]".format(st, st + data_length))
        st += data_length - overlap
    print("[{}, {}]".format(length - data_length - 1, length - 1))


# def find_wavlength():
#     w2m = Wave2Mel(sr=22050)
#     for length in range(22050, 33075, 1000):
#         print(w2m(torch.rand(4, length)).shape)


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


def merge_segments(segments):
    pass


def split_series(series):
    pass


def wav_fold():
    bs = 16
    length = 19
    sig_length = 22050
    overlap = sig_length // 2
    x_list = [torch.rand(sig_length) for _ in range(length)]
    wav_inp = None
    for item in x_list:
        if wav_inp is None:
            wav_inp = item
        else:
            wav_inp = torch.concat((wav_inp[:len(wav_inp) - overlap], item), dim=-1)
    print("overlap:{}, length:{}".format(overlap, wav_inp.shape))

    y, sr = np.random.rand(367543), 22050
    print("wav length:", y.shape)
    data_length, seg_length = sr, 19
    overlap = data_length // 2
    series_length = data_length + overlap*(seg_length-1)
    print("series length:", series_length)


def split_image():
    from PIL import Image
    img = np.array(Image.open("C:/Users/zhaoke/Documents/paper1/waveform/fullwaveform.png"))

    print(img.shape)
    st, step = 0, 260
    cnt = 0
    while st<2000:
        tmp = img[:, st:st+step, :]
        pil_img = Image.fromarray(tmp.astype(np.uint8))
        pil_img.save("C:/Users/zhaoke/Documents/paper1/waveform/seg_{}.png".format(cnt))
        st = st+step
        cnt += 1
        if cnt == 7:
            break


if __name__ == '__main__':
    split_image()
    # wav_fold()

    # sample_list, label_list, noise_list = get_combined_data()
    # # trte_rate = int(len(sample_list) * 0.9)
    # train_loader = DataLoader(
    #     CoughDataset(audioseg=sample_list, labellist=label_list, noises=noise_list),
    #     batch_size=64, shuffle=True)
    # for batch_id, (x_wav, y_lab) in tqdm(enumerate(train_loader),
    #                                      desc="Training "):
    #     x_wav = x_wav.unsqueeze(1)
    #     print(x_wav.shape, y_lab.shape)
