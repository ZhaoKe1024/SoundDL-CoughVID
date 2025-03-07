# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2025-02-15 1:25
import json
import os
import random

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


def min2sec(t: str):
    parts = t.split(':')
    res = float(parts[-1])
    f = 60
    for i in range(len(parts) - 1):
        res += int(parts[len(parts) - 2 - i]) * f
        f *= 60
    return res


def min2sec1(t: str) -> float:
    """将"MM:ss.ss"格式的时间字符串转换为总秒数（浮点数）。"""
    minutes_str, seconds_str = t.split(':')
    minutes = int(minutes_str)
    seconds = float(seconds_str)
    return minutes * 60 + seconds


def sec2min(f: float) -> str:
    """将总秒数（浮点数）转换为"MM:ss.ss"格式的时间字符串，支持负数。"""
    sign = -1 if f < 0 else 1
    f_abs = abs(f)
    minutes = int(f_abs // 60)
    seconds = f_abs % 60
    # 四舍五入到两位小数并处理进位
    seconds_rounded = round(seconds, 2)
    if seconds_rounded >= 60:
        minutes += 1
        seconds_rounded -= 60
    # 格式化为两位整数分钟和两位整数+两位小数秒
    minutes_str = f"{minutes:02d}"
    seconds_str = f"{seconds_rounded:05.2f}"  # 例如 5.5 → 05.50
    time_str = f"{minutes_str}:{seconds_str}"
    return f"-{time_str}" if sign < 0 else time_str


def s2p(sec):
    return int(min2sec1(sec) * 22050)


def subtract(t1: str, t2: str) -> str:
    """计算两个时间字符串的差值，返回"MM:ss.ss"格式的结果。"""
    diff = min2sec1(t1) - min2sec1(t2)
    return sec2min(diff)


class BiliCoughReader(object):
    def __init__(self, ROOT="G:/DATAS-Medical/BILIBILICOUGH/", task="scd"):
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

        # ==========---------scd------------============
        self.NOISE_ROOT = "G:/DATAS-Medical/BILINOISE/"
        filter_length = 25
        self.noise_files = ["bilinoise_01.wav", "bilinoise_02.wav"]
        self.noise_length = [20638720, 39693293]
        # for item in os.listdir(self.NOISE_ROOT):
        #     if item[-4:] == ".wav" and len(item) >= filter_length:
        #         self.noise_files.append(item)
        if task == "scd_generate":
            self.scdfile_path = "G:/DATAS-Medical/BILIBILICOUGH/bilicough_metainfo_c4scd_250307_dura.csv"
            self.scd_df = pd.read_csv(self.scdfile_path, delimiter=',', header=0, index_col=0,
                                      usecols=[0, 1, 2, 3, 5], encoding="ansi")
            self.target_samples = int(3.0 * 22050)
            self.gap_threshold = int(1.5 * 22050)
            self.sequences = []
            self.current_seq = None

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

    def get_multi_event_batches(self, filepath):
        json_str = None  # json string
        if filepath is None:
            filepath = "../datasets/metainfo4scd.json"
        with open(filepath, 'r', encoding='utf_8') as fp:
            json_str = fp.read()
        json_data = json.loads(json_str)
        segments, labels = [], []
        y, sr = None, None
        yname = None
        yn1, sr = librosa.load("G:/DATAS-Medical/BILINOISE/bilinoise_01.wav")
        yn2, sr = librosa.load("G:/DATAS-Medical/BILINOISE/bilinoise_02.wav")
        for key in tqdm(json_data, desc="Loading.."):
            # for key in json_data:
            fname = json_data[key]["filename"]
            if (yname is None) or (fname != yname):
                yname = fname + ".wav"
                y, sr = librosa.load("G:/DATAS-Medical/BILIBILICOUGH/" + yname)
            item_segs = json_data[key]["segments"]

            st, en = json_data[key]["start"], None
            nfile, nst, nen = None, None, None
            if item_segs[-1]["filename"][:9] == "bilinoise":
                # st =
                en = item_segs[-2]["en"]
            else:
                # st = json_data[key]["start"]
                en = json_data[key]["end"]
            if type(st) == str:
                st = s2p(st)
            if type(en) == str:
                en = s2p(en)
            # print(st, type(st), en, type(en))
            seg, label = y[st:en], json_data[key]["labels"]

            if item_segs[-1]["filename"][:9] == "bilinoise":
                # print("add noise!")
                nfile = item_segs[-1]["filename"]
                nst, nen = item_segs[-1]["st"], item_segs[-1]["en"]
                # print("nfile:", nfile)
                if type(nst) == str:
                    nst = s2p(nst)
                if type(nen) == str:
                    nen = s2p(nen)
                # print(nen, type(nen), nst, type(nst), en, type(en), st, type(st))
                seglen = nen-nst+len(seg)  # 这里nen-nst+len(seg)和nen-nst+en-st的结果不同！可见seg本身可能长度就不够的。
                # print(seglen)
                if seglen < 66150:
                    # print("Error!!!!!")
                    nen += 66150 - seglen
                if nfile == "bilinoise_01.wav":
                    yn = yn1[nst:nen]
                    # print("seg, len:", len(seg), len(yn))
                    seg = np.concatenate((seg, yn), axis=0)
                    # print("seg:", len(seg))
                elif nfile == "bilinoise_02.wav":
                    yn = yn2[nst:nen]
                    # print("seg, len:", len(seg), len(yn))
                    seg = np.concatenate((seg, yn), axis=0)
                    # print("seg:", len(seg))
                else:
                    raise ValueError("Unknown Noise Filename.")

            if len(seg) != 22050 * 3:
                errer_massage = "Error Length:"+str(len(seg))+","+fname
                errer_massage += ", isnoised:"+str(item_segs[-1]["filename"][:9])
                errer_massage += ", en:{}, st:{}, {}, nen:{}, nst:{}".format(en, st, en-st, nen, nst)
                if nen is not None:
                    errer_massage += ", "+str(nen-nst)
                raise ValueError(errer_massage)
            segments.append(seg)
            labels.append(label)
        return segments, labels
            # print(key, ":", fname, st, en, nfile, nst, nen)

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

    def __add_noise_segment(self, remaining_duration: float) -> dict:
        """添加噪音片段"""
        # 随机选择噪音文件并加载（此处需要实现实际音频加载逻辑）
        noise_id = np.random.randint(1, 2)
        # 假设实现获取指定时长的噪音片段的逻辑
        y, sr = librosa.load(self.NOISE_ROOT + self.noise_files[noise_id])
        st = random.randint(0, self.noise_length[noise_id] - int(remaining_duration))

        return {
            'filename': self.noise_files[noise_id],
            'st': st,
            'en': st + remaining_duration,
            'd_label': 0
        }

    def __finalize_sequence(self):
        """完成当前序列构建"""
        if self.current_seq and len(self.current_seq['segments']) > 0:
            # 计算总持续时间
            total_dur = self.current_seq['end'] - self.current_seq['start']

            # 处理不足时长的情况
            if total_dur < self.target_samples:
                needed = self.target_samples - total_dur
                noise_segment = self.__add_noise_segment(needed)
                self.current_seq['segments'].append(noise_segment)
                self.current_seq['end'] += needed

            diseases = [s['d_label'] for s in self.current_seq['segments']]
            labels = [0, 0, 0, 0]
            # 这里有个问题，我没判断是否具有多个标签
            if 1 in diseases:
                labels[1] += 1
            elif 2 in diseases:
                labels[2] += 1
            elif 3 in diseases:
                labels[3] += 1
            if labels[1] + labels[2] + labels[3] == 0:
                label = 0
            elif labels[1] > 0:
                if labels[2] > 0 or labels[3] > 0:
                    print("-------------------multi-label 1, 2 or 3!!!!!!!!!!!")
                label = 1
            elif labels[2] > 0:
                if labels[3] > 0:
                    print("-------------------multi-label 2 3!!!!!!!!!!!")
                label = 2
            elif labels[3] > 0:
                label = 3
            else:
                raise ValueError("Unknown reason that the label is None.")
            # 添加到最终序列
            # 其实这里忽略了噪声数据，后续我改改
            self.sequences.append({
                "filename": self.current_seq['filename'],
                "start": self.current_seq['start'],
                "end": self.current_seq['start'] + self.target_samples,
                "labels": label,
                "segments": self.current_seq["segments"]
            })

            # 处理截断残留（如果存在）
            last_segment = self.current_seq['segments'][-1]
            if type(last_segment['en']) == str:
                last_segment['en'] = s2p(last_segment['en'])
            # print(last_segment['en'], self.sequences[-1]['end'])
            # print(last_segment)
            if last_segment['en'] > self.sequences[-1]['end']:
                residual = {
                    "filename": last_segment['filename'],
                    "st": self.sequences[-1]['end'],
                    "en": last_segment['en'],
                    "d_label": last_segment['d_label']
                }
                self.current_seq = {
                    "filename": self.current_seq['filename'],
                    "start": residual['st'],
                    "end": residual['en'],
                    "segments": [residual]
                }
            else:
                self.current_seq = None

    def genarate_series_batches(self):
        """主处理流程
            bcr = BiliCoughReader()
            seqs = bcr.get_multi_event_batches()
            for i in range(len(seqs)):
                print('\"item{}\":'.format(i), seqs[i], ',')
        """
        # 按文件分组处理
        for fname, group in self.scd_df.groupby('filename'):
            self.current_seq = None  # 重置当前序列
            # print("now filename:", fname)
            for _, row in tqdm(group.iterrows(), desc="Now {}".format(fname)):
                if row['d_label'] > 3:
                    continue
                # 初始化新序列
                if self.current_seq is None:
                    self.current_seq = {
                        "filename": fname,
                        "start": s2p(row['st']),
                        "end": s2p(row['en']),
                        "segments": [row.to_dict()]
                    }
                    continue

                # 计算时间间隔
                gap = s2p(row['st']) - self.current_seq['end']
                # min2sec1(t1) - min2sec1(t2)
                # 判断是否合并
                if gap <= self.gap_threshold:
                    # 合并到当前序列
                    self.current_seq['end'] = s2p(row['en'])
                    self.current_seq['segments'].append(row.to_dict())

                    # 检查是否超长
                    if (self.current_seq['end'] - self.current_seq['start']) >= self.target_samples:
                        self.__finalize_sequence()
                else:
                    # 结束当前序列并创建新序列
                    self.__finalize_sequence()
                    self.current_seq = {
                        "filename": fname,
                        "start": s2p(row['st']),
                        "end": s2p(row['en']),
                        "segments": [row.to_dict()]
                    }

            # 处理文件末尾的残留序列
            if self.current_seq is not None:
                self.__finalize_sequence()
        return self.sequences


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

    # strlist = ["bilicough_031,,WET_COUGH_VS_DRY_COUGH_Hear_the_Difference,male,adult",
    #            "bilicough_032,,Have_you_heard_this_cough_before,female,child",
    #            "bilicough_033,,Types_of_Coughs_in_60_Sec,male,adult",
    #            "bilicough_034,,Dry_cough_sound_effect,male,adult",
    #            "bilicough_035,,Smokers_Coughing_SOUND_EFFECT-Unhealthy_Cough_Ungesund_Raucherhusten_SOUNDS,male,adult",
    #            "bilicough_036,,Some_wet_and_barking_coughing",
    #            "bilicough_037,,Whooping_Cough_in_an_Adult",
    #            "bilicough_038,,asthma"]
    # kvlist = []
    # ind = 31
    # for stritem in strlist:
    #     parts = stritem.split(',')
    #     kvlist.append((parts[2], "bilicough_0{}".format(ind)))
    #     ind += 1
    # ffmpeg_mp42wav_list(kvlist)


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
    # generate_SCD_metainfo(task="sed")  # generate metainfo.csv
    # y, sr = librosa.load("G:/DATAS-Medical/BILINOISE/bilinoise_01.wav")
    # print(len(y))
    # y, sr = librosa.load("G:/DATAS-Medical/BILINOISE/bilinoise_02.wav")
    # print(len(y))
    bcr = BiliCoughReader()
    bcr.get_multi_event_batches()
    # print(seqs[10])
    # ncr = NEUCoughReader()
    # cvr = CoughVIDReader()
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
