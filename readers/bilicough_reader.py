# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2025-02-15 1:25
import os

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

        # 该字典是bilicough数据集用的，原理是按照字母排序，不能修改！
        # sed_name2label = {"breathe": 0, "cough": 2, "clearthroat": 1, "exhale": 3, "hum": 4, "inhale": 5, "noise": 6, "silence": 7,
        #               "sniff": 8, "speech": 9, "vomit": 10, "whooping": 11}

        # 这个字典是去掉silence和noise之后的上述字典，分类任务用的，也不能修改！
        self.sed_label2name = {0: "breathe", 1: "clearthroat", 2: "cough", 3: "exhale", 4: "hum", 5: "inhale",
                               6: "sniff", 7: "speech", 8: "vomit", 9: "whooping"}
        # self.vad_name2label = {"silence or noise": 0, "sound activity": 1}
        self.vad_label2name = {0: "silence or noise", 1: "sound activity"}

    def get_sample_label_list(self, mode):
        """The Only Function to be Called.
        event:
            sed: 0,1,2,...,11, all sound events.
            vad: 0-1 labelled.
            disease: asthma and so on.
            """
        if mode == "vad":
            return self.__get_vad_data()
        elif mode == "sed":
            return self.__get_sed_data()
        elif mode == "scd":
            return self.get_multi_event_batches()
        else:
            raise Exception("Unknown param event: {} !!!!".format(mode))

    def __get_sed_data(self):
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
        sr = None
        # filename	st	en	labelfull	labelname	label	binlab
        for ind, item in enumerate(metadf.itertuples()):
            if (cur_fname != item[1]) or (cur_fname is None):
                cur_fname = item[1]
                cur_wav, sr = librosa.load(self.ROOT + cur_fname + ".wav")
                if sr not in sr_list:
                    sr_list.append(sr)
                data_length = sr
                # print("BBilicough Data length:", data_length)
            st, en = int(min2sec(item[2]) * sr), int(min2sec(item[3]) * sr + 1)
            if en > len(cur_wav):
                en = len(cur_wav)
            if en - st < 100:
                raise Exception("Error Index.")
            label = int(item[4])
            sn = en - st
            # sec = (en - st)/22050
            if (pre_en is None):
                if st >= data_length:
                    st_pos = 0
                    ind = 0
                    while st_pos + data_length <= st:
                        st_pos += data_length
                        ind += 1
                        if ind > 2:
                            break
            else:
                if st - pre_en >= sr:
                    st_pos = pre_en
                    ind = 0
                    while st_pos + data_length <= st:
                        st_pos += data_length
                        ind += 1
                        if ind > 2:
                            break
            if sn == data_length:
                # if len(cur_wav[st:en]) != sr:
                #     raise Exception("Error Length.")
                if label not in [6, 7]:
                    sample_list.append(cur_wav[st:en])
                    label_list.append(label)
            elif sn < data_length:
                new_sample = np.zeros(data_length)
                # print(st, en, sn, len(cur_wav), item[1])
                if en <= len(cur_wav):
                    new_sample[:sn] = cur_wav[st:en]
                else:
                    new_sample[:sn] = cur_wav[len(cur_wav) - sn:len(cur_wav)]
                # if len(new_sample) != sr:
                #     raise Exception("Error Length.")
                if label not in [6, 7]:
                    sample_list.append(new_sample)
                    label_list.append(label)
            else:
                cnt_sum = sn // data_length + 1
                res = cnt_sum * data_length - sn
                overlap = res // (cnt_sum - 1)
                st_pos = st
                while st_pos + data_length < en:
                    if label not in [6, 7]:
                        sample_list.append(cur_wav[st_pos:st_pos + data_length])
                        label_list.append(label)
                    st_pos += data_length - overlap
                if label not in [6, 7]:
                    sample_list.append(cur_wav[en - data_length:en])
                    label_list.append(label)
            pre_st, pre_en = st, en
        print("sound count:{}, all count:{}.".format(sum(label_list), len(label_list)))
        print(sr_list)
        self.sr = sr_list[0]
        self.data_length = data_length
        # name2label = {"breathe": 0, "cough": 2, "clearthroat": 1, "exhale": 3, "hum": 4, "inhale": 5, "noise": 6, "silence": 7,
        #               "sniff": 8, "speech": 9, "vomit": 10, "whooping": 11}
        # Since labels 6 and 7 represent noise and mute respectively, which have been filtered out in the previous step,
        # it is necessary to squeeze labels 8, 9, 10 and 11 to subtract 2
        final_label = []
        for it in label_list:
            if it > 7:
                final_label.append(it - 2)
            elif it < 6:
                final_label.append(it)
            else:
                raise ValueError("Labels 6 and 7 cannot appear.")
        return sample_list, final_label

    def __get_vad_data(self):
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
        sr = None
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

    def get_multi_event_batches(self):
        # 还没给疾病标签标完呢！
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
        sr = None
        # filename	st	en	labelfull	labelname	label	binlab
        for ind, item in enumerate(metadf.itertuples()):
            if (cur_fname != item[1]) or (cur_fname is None):
                cur_fname = item[1]
                cur_wav, sr = librosa.load(self.ROOT + cur_fname + ".wav")
                if sr not in sr_list:
                    sr_list.append(sr)
                data_length = sr
                # print("BBilicough Data length:", data_length)
            st, en = int(min2sec(item[2]) * sr), int(min2sec(item[3]) * sr + 1)
            if en > len(cur_wav):
                en = len(cur_wav)
            if en - st < 100:
                raise Exception("Error Index.")
            label = int(item[4])
            sn = en - st
            # sec = (en - st)/22050
            if (pre_en is None):
                if st >= data_length:
                    st_pos = 0
                    ind = 0
                    while st_pos + data_length <= st:
                        st_pos += data_length
                        ind += 1
                        if ind > 2:
                            break
            else:
                if st - pre_en >= sr:
                    st_pos = pre_en
                    ind = 0
                    while st_pos + data_length <= st:
                        st_pos += data_length
                        ind += 1
                        if ind > 2:
                            break
            if sn == data_length:
                # if len(cur_wav[st:en]) != sr:
                #     raise Exception("Error Length.")
                if label not in [6, 7]:
                    sample_list.append(cur_wav[st:en])
                    label_list.append(label)
            elif sn < data_length:
                new_sample = np.zeros(data_length)
                # print(st, en, sn, len(cur_wav), item[1])
                if en <= len(cur_wav):
                    new_sample[:sn] = cur_wav[st:en]
                else:
                    new_sample[:sn] = cur_wav[len(cur_wav) - sn:len(cur_wav)]
                # if len(new_sample) != sr:
                #     raise Exception("Error Length.")
                if label not in [6, 7]:
                    sample_list.append(new_sample)
                    label_list.append(label)
            else:
                cnt_sum = sn // data_length + 1
                res = cnt_sum * data_length - sn
                overlap = res // (cnt_sum - 1)
                st_pos = st
                while st_pos + data_length < en:
                    if label not in [6, 7]:
                        sample_list.append(cur_wav[st_pos:st_pos + data_length])
                        label_list.append(label)
                    st_pos += data_length - overlap
                if label not in [6, 7]:
                    sample_list.append(cur_wav[en - data_length:en])
                    label_list.append(label)
            pre_st, pre_en = st, en
        print("sound count:{}, all count:{}.".format(sum(label_list), len(label_list)))
        print(sr_list)
        self.sr = sr_list[0]
        self.data_length = data_length
        # name2label = {"breathe": 0, "cough": 2, "clearthroat": 1, "exhale": 3, "hum": 4, "inhale": 5, "noise": 6, "silence": 7,
        #               "sniff": 8, "speech": 9, "vomit": 10, "whooping": 11}
        # Since labels 6 and 7 represent noise and mute respectively, which have been filtered out in the previous step,
        # it is necessary to squeeze labels 8, 9, 10 and 11 to subtract 2
        final_label = []
        for it in label_list:
            if it > 7:
                final_label.append(it - 2)
            elif it < 6:
                final_label.append(it)
            else:
                raise ValueError("Labels 6 and 7 cannot appear.")
        return sample_list, final_label


def add_the_disease_label():
    # metadf = pd.read_csv("G:/DATAS-Medical/BILIBILICOUGH/bilicough_metainfo.csv", delimiter=',',
    #                      header=0, index_col=None, usecols=[0, 1, 2, 5], encoding="ansi")
    # print(metadf)
    # cur_fname = None
    # cur_wav = None
    # data_length = None
    # sample_list = []
    # label_list = []
    # sr_list = []
    # pre_st, pre_en = None, None
    # sr = None
    # # filename	st	en	labelfull	labelname	label	binlab
    # for ind, item in enumerate(metadf.itertuples()):
    #     if (cur_fname != item[1]) or (cur_fname is None):
    #         cur_fname = item[1]
    for item in os.listdir("G:/DATAS-Medical/BILIBILICOUGH/"):
        if item[-3:] == "ass":
            assfname = "G:/DATAS-Medical/BILIBILICOUGH/" + item
            print(assfname)
            fin = open(assfname, 'r', encoding="utf_8")
            line = fin.readline()
            while line:
                if line[:8] != "[Events]":
                    line = fin.readline()
                else:
                    break
            fin.readline()
            line = fin.readline()
            while line:
                parts = line.split(',')
                parts1 = parts[9].split('(')
                if len(parts1) == 2:
                    print(parts1[1][:-2])
                line = fin.readline()
            # print(line)
            fin.close()


if __name__ == '__main__':
    add_the_disease_label()
