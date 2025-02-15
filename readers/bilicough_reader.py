# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2025-02-15 1:25
import numpy as np
import pandas as pd
import librosa


def min2sec(t: str):
    parts = t.split(':')
    res = float(parts[-1])
    f = 60
    for i in range(len(parts) - 1):
        res += int(parts[len(parts) - 2 - i]) * f
        f *= 60
    return res


class BiliCoughReader(object):
    def __init__(self, ROOT="G:/DATAS-Medical/BILIBILICOUGH/"):
        self.ROOT = ROOT
        # metadf = pd.read_csv(ROOT + "bilicough_metainfo.csv", delimiter=',', header=0,
        #                      index_col=None, encoding="ansi")
        self.sr = None
        self.data_length = None

    def get_sample_label_list(self):
        """The Only Function to be Called."""
        metadf = pd.read_csv(self.ROOT + "bilicough_metainfo.csv", delimiter=',', header=0, index_col=None,
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
                cur_wav, sr = librosa.load(self.ROOT + cur_fname + ".wav")
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
        print(sr_list)
        self.sr = sr_list[0]
        self.data_length = data_length
        return sample_list, label_list
