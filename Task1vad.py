#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/1/2 20:29
# @Author: ZhaoKe
# @File : Task1vad.py
# @Software: PyCharm
import os
import numpy as np
import pandas as pd
import librosa


# newdf.groupby("binlab").count()
def min2sec(t: str):
    parts = t.split(':')
    res = float(parts[-1])
    f = 60
    for i in range(len(parts) - 1):
        res += int(parts[len(parts) - 2 - i]) * f
        f *= 60
    return res


def load_bilicough_dataset():
    ROOT = "G:/DATAS-Medical/BILIBILICOUGH/"
    metadf = pd.read_csv(ROOT + "bilicough_metainfo.csv", delimiter=',', header=0, index_col=None, usecols=[0, 1, 2, 5],
                         encoding="ansi")
    print(metadf)
    cur_fname = None
    cur_wav = None
    data_length = None
    sample_list = []
    label_list = []
    sr_list = []
    sr = None
    pre_st, pre_en = None, None
    # filename	st	en	labelfull	labelname	label	binlab
    for ind, item in enumerate(metadf.itertuples()):
        if (cur_fname != item[1]) or (cur_fname is None):
            cur_fname = item[1]
            cur_wav, sr = librosa.load(ROOT + cur_fname + ".wav")
            if sr not in sr_list:
                sr_list.append(sr)
            data_length = int(sr)
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
    print(sr_list)
    return sample_list, label_list


def load_bilinoise_dataset():
    NOISE_ROOT = "G:/DATAS-Medical/BILINOISE/"
    noise_length = None
    filter_length = 25
    ind = 0
    new_noise_list = []
    for item in os.listdir(NOISE_ROOT):
        if item[-4:] == ".wav" and len(item) >= filter_length:
            cur_fname = NOISE_ROOT + item
            cur_wav, sr = librosa.load(cur_fname)
            noise_length = int(sr)
            L = len(cur_wav)
            st_pos = np.random.randint(0, L - noise_length)
            new_noise_list.append(cur_wav[st_pos:st_pos + noise_length])
            # print(NOISE_ROOT+item)
        ind += 1
        if ind > 18:
            break
    for item in new_noise_list:
        print(len(item))


if __name__ == '__main__':
    sample_list, label_list = load_bilicough_dataset()
    length_list = []
    for item in sample_list:
        if len(item) not in length_list:
            length_list.append(len(item))
    print(length_list)
