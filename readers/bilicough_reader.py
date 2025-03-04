# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2025-02-15 1:25
import json
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
        metadf = pd.read_csv(self.ROOT + "bilicough_metainfo_c4scd.csv", delimiter=',', header=0, index_col=None,
                             usecols=[0, 1, 2, 6], encoding="ansi")
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
            label = int(item[4])
            if label not in [0, 1, 2, 3, 4, 5]:
                continue

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
                if label not in [0]:
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
                if label not in [0]:
                    sample_list.append(new_sample)
                    label_list.append(label)
            else:
                # cnt_sum = sn // data_length + 1
                # res = cnt_sum * data_length - sn
                # overlap = res // (cnt_sum - 1)

                overlap = data_length // 2
                st_pos = st
                while st_pos + data_length < en:
                    if label not in [0]:
                        sample_list.append(cur_wav[st_pos:st_pos + data_length])
                        label_list.append(label)
                    st_pos += data_length - overlap
                if label not in [0]:
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
        # final_label = []
        # for it in label_list:
        #     if it > 7:
        #         final_label.append(it - 2)
        #     elif it < 6:
        #         final_label.append(it)
        #     else:
        #         raise ValueError("Labels 6 and 7 cannot appear.")
        return sample_list, label_list

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


def get_filelist():
    ROOT = "F:/DATAS/bilicough250226/"
    idx = 19
    for item in os.listdir(ROOT):
        if item[-3:] == "mp4":
            print("bilicough_{},,{}".format(("00" + str(idx))[-3:], item[:-3]))
            idx += 1


def ffmpeg_mp42wav(root_path):
    for item in os.listdir(root_path):
        if item[-3:] == "mp4":
            name = item.split('.')[0]
            print("ffmpeg -i {}.mp4 -f wav -ar 22050 {}.wav".format(root_path + name, root_path + name))
            os.system("ffmpeg -i {}.mp4 -f wav -ar 22050 {}.wav".format(root_path + name, root_path + name))


def ffmpeg_mp42wav_list(kv_list):
    ROOT = "G:/DATAS-Medical/BILIBILICOUGH/"
    for (mp4name, wavname) in kv_list:
        print("ffmpeg -i {}.mp4 -f wav -ar 22050 {}.wav".format(ROOT + mp4name, ROOT + wavname))
        os.system("ffmpeg -i {}.mp4 -f wav -ar 22050 {}.wav".format(ROOT + mp4name, ROOT + wavname))


def generate_SCD_metainfo(task="scd"):
    """在BiliCough中添加新的数据
    给定一条长音频，我手动在Aegisub软件中标注好ass文件，然后通过这个函数去简易地切分为目标长度的短片段"""
    root_path = "G:/DATAS-Medical/BILIBILICOUGH/"
    # scd_indices = [1, 4, 7, 8, 9, 10, 11, 14, 15, 20, 25, 28]
    # d_labels = ["whooping", "asthma", "pneumonia"]
    if task == "scd":
        d_dict = {"000": "healthy",
                  "001": "whooping", "003": "healthy", "004": "whooping", "007": "asthma", "008": "asthma",
                  "009": "pneumonia", "010": "whooping",
                  "011": "whooping",
                  "014": "whooping", "015": "asthma", "018": "whooping", "019": "healthy", "020": "pneumonia",
                  "021": "healthy",
                  "022": "healthy", "025": "COPD", "028": "whooping", "033": "", "037": "whooping", "038": "asthma"
                  }
        fname = "c4scd"
    elif task == "sed":
        d_dict = [("00" + str(it))[-3:] for it in range(39)]
        fname = "c2sed"
    else:
        raise ValueError("Unknown task:{}.".format(task))
    metainfo_file = open("G:/DATAS-Medical/BILIBILICOUGH/bilicough_metainfo_{}.csv".format(fname), 'w')
    metainfo_file.write("filename,st,en,disease,d_label,event,e_label\n")
    json_str = None  # json string
    with open("../configs/ucaslabel.json", 'r', encoding='utf_8') as fp:
        json_str = fp.read()
    json_data = json.loads(json_str)
    disease2label = json_data["disease2label"]
    event2label = json_data["event2label"]
    label_cnt0 = dict()
    label_cnt1 = dict()
    for key in d_dict:
        wname = root_path + "bilicough_" + key
        print("--------------->file name:", wname)
        # wavtest = fname + ".wav"
        asstest = wname + ".ass"

        # label_names = ["breathe", "cough", "clearthroat", "exhale", "hum", "inhale", "noise", "silence", "sniff",
        #                "speech",
        #                "vomit", "whooping"]

        assfin = open(asstest, 'r', encoding="utf-8")
        # label_list = []
        line = assfin.readline()
        while line.strip() != "[Events]":
            line = assfin.readline()
            # print(line)
        assfin.readline()
        line = assfin.readline()
        while line:
            # print(line)
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
            if e_label in ["breathe", "wheeze"]:
                raise ValueError("Error key:{}".format(e_label))
            if e_label in ["useless", "music"]:
                line = assfin.readline()
                continue
            else:
                # if e_label == "breathe":
                #     print(asstest)
                assert e_label in event2label, "Unknown sound event label:{}".format(e_label)
                assert d_label in disease2label, "Unknown disease label:{}".format(d_label)
                if e_label not in label_cnt1:
                    label_cnt1[e_label] = 1
                else:
                    label_cnt1[e_label] = label_cnt1.get(e_label) + 1

                if d_label not in label_cnt0:
                    label_cnt0[d_label] = 1
                else:
                    label_cnt0[d_label] = label_cnt0.get(d_label) + 1

                    print("{},{},{},{},{},{}\n".format("bilicough_" + key, parts[1], parts[2], d_label,
                                                       disease2label[d_label], e_label, event2label[e_label]))
                    metainfo_file.write(
                        "{},{},{},{},{},{},{}\n".format("bilicough_" + key, parts[1], parts[2], d_label,
                                                        disease2label[d_label], e_label, event2label[e_label]))
                line = assfin.readline()

        # for item in label_list:
        #     print(item)
        assfin.close()

    print("标签分布：")
    for k, v in label_cnt0.items():
        print("key:{},\tcount:{}".format(k, v))
    print("---------------=============----------------")
    for k, v in label_cnt1.items():
        print("key:{},\tcount:{}".format(k, v))
    print("运行完毕，数据已写入到bilicough_metainfo_{}.csv里面".format(fname))
    metainfo_file.close()


if __name__ == '__main__':
    strlist = ["bilicough_031,,WET_COUGH_VS_DRY_COUGH_Hear_the_Difference,male,adult",
               "bilicough_032,,Have_you_heard_this_cough_before,female,child",
               "bilicough_033,,Types_of_Coughs_in_60_Sec,male,adult",
               "bilicough_034,,Dry_cough_sound_effect,male,adult",
               "bilicough_035,,Smokers_Coughing_SOUND_EFFECT-Unhealthy_Cough_Ungesund_Raucherhusten_SOUNDS,male,adult",
               "bilicough_036,,Some_wet_and_barking_coughing",
               "bilicough_037,,Whooping_Cough_in_an_Adult",
               "bilicough_038,,asthma"]
    kvlist = []
    ind = 31
    for stritem in strlist:
        parts = stritem.split(',')
        kvlist.append((parts[2], "bilicough_0{}".format(ind)))
        ind += 1
    ffmpeg_mp42wav_list(kvlist)

    # generate_SCD_metainfo(task="sed")

    # bcr = BiliCoughReader()
    # # ncr = NEUCoughReader()
    # # cvr = CoughVIDReader()
    # sample_list, label_list = [], []
    # tmp_sl, tmp_ll = bcr.get_sample_label_list(mode="sed")
    # sample_list.extend(tmp_sl)
    # label_list.extend(tmp_ll)
    # print("bilicough:", len(label_list), bcr.data_length)

    # add_the_disease_label()
    # get_filelist()
    # ffmpeg_mp42wav(root_path="F:/DATAS/bilicough250226/")
    # root_path = "F:/DATAS/bilicough250226/"
    # name = "你们想听到的犬吠样咳嗽它来了"
    # os.system("ffmpeg -i {}.mp4 -f wav -ar 22050 {}.wav".format(root_path + name, root_path + name))
    # ROOT = "F:/DATAS/bilicough250226/"
    # # idx = 19
    # for item in os.listdir(ROOT):
    #     if item[-3:] == "wav":
    #         y, sr = librosa.load(ROOT + item)
    #         print(sr, ':', len(y))
    #         # idx += 1
