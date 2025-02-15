#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/11/14 22:02
# @Author: ZhaoKe
# @File : neucough_reader.py
# @Software: PyCharm
# import sys
# sys.path.append(r'C:/Program Files (zk)/PythonFiles/AClassification/SoundDL-CoughVID')
import random
import numpy as np
import librosa
from tqdm import tqdm


def min2sec(t: str):
    parts = t.split(':')
    return int(parts[0]) * 60 + float(parts[1])


def wav_slice_padding(old_signal, save_len=22050, usenoise=False):
    new_signal = None
    if old_signal.shape[0] < save_len:
        if usenoise and random.random() < 0.333:
            raise Exception("This place is temporarily empty. Please set parameter usenoise to false.")
        else:
            new_signal = np.zeros(save_len)
        resi = save_len - old_signal.shape[0]
        # print("resi:", resi)
        new_signal[:old_signal.shape[0]] = old_signal
        new_signal[old_signal.shape[0]:] = old_signal[-resi:][::-1]
    elif old_signal.shape[0] > save_len:
        posi = random.randint(0, old_signal.shape[0] - save_len)
        new_signal = old_signal[posi:posi + save_len]
    return new_signal


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


class NEUCoughReader(object):
    def __init__(self):
        # , ROOT="F:/DATAS/NEUCOUGHDATA_COUGHSINGLE/"
        self.ROOT = "F:/DATAS/NEUCOUGHDATA_FULL/"
        self.metapath = "F:/DATAS/NEUCOUGHDATA_COUGHSINGLE/neucough_metainfo.txt"
        # metadf = pd.read_csv(ROOT + "bilicough_metainfo.csv", delimiter=',', header=0,
        #                      index_col=None, encoding="ansi")
        self.sr = None
        self.data_length = None
        self.desc = ""

    def get_sample_label_list(self, mode="cough"):
        """

        :param event:[cough: only cough event, all: all sound event]
        :return:
            sample_list: list of data.
            lael_list: list of label indicating the cough(2) and speech(9), silence(7).
        """
        sample_list, label_list = [], []
        if mode == "cough":
            fin = open(self.metapath, 'r')
            fin.readline()
            line = fin.readline()
            curfile = None
            w_data, sr = None, None
            pbar = tqdm(total=321)
            while line:
                parts = line.split(',')
                fname = parts[0]
                if (fname != curfile) or (curfile is None):
                    curfile = fname
                    w_data, sr = librosa.load(path=self.ROOT + "{}_audiodata_元音字母a.wav".format(curfile))
                    if self.sr is None:
                        self.sr = sr
                        self.data_length = sr
                st, en = int(min2sec(parts[1]) * sr), int(min2sec(parts[2]) * sr + 1)
                one_data = w_data[st:en]
                segs = None
                if len(one_data) > self.data_length:
                    segs = self.__split(one_data)
                elif len(one_data) < self.data_length:
                    segs = self.__padding(one_data)
                else:
                    segs = [one_data]
                sample_list.extend(segs)
                label_list.extend([2] * len(segs))
                line = fin.readline()
                pbar.update(1)
            fin.close()
        elif mode == "all":
            pass
        else:
            raise ValueError("Unknown param event: {} !!!!".format(mode))
        return sample_list, label_list

    def __split(self, w_data):
        L = w_data.shape[0]
        overlap = int(self.data_length // 6)
        if L - self.data_length < overlap:
            return [w_data]
        else:
            segs = []
            st = 0
            while st + self.data_length < L:
                segs.append(w_data[st:st + self.data_length])
                st = st + self.data_length - overlap
            if st + self.data_length - L < overlap:
                segs.extend(self.__padding(w_data[st:]))
            return segs

    def __padding(self, w_data, usenoise=False):
        new_signal = None
        if usenoise:
            pass
        else:
            new_signal = np.zeros(self.data_length)
            # resi = self.data_length - w_data.shape[0]
            # print("resi:", resi)
            new_signal[:w_data.shape[0]] = w_data
            new_signal[-w_data.shape[0]:] = w_data[::-1]
        return [new_signal]

    def __read_Cough_item(self, fname):
        # It doesn't seem to be available for the time being.

        pass

    def __read_SoundEvent_item(self, wavname):
        """The Only Function to be Called."""
        intervals = read_ass_to_segs(self.ROOT + "{}_annotations.ass".format(wavname))
        print(intervals)
        sample, sr = librosa.load(self.ROOT + "{}_audiodata_元音字母a.wav".format(wavname))
        step, overlap = 8000, 2000
        st_pos, en_pos = 0, step
        st_pre, en_pre = -6001, -1
        Length = len(sample)
        st_cur, en_cur = int(intervals[0][0] * sr), int(intervals[0][1] * sr)
        st_tail, en_tail = int(intervals[1][0] * sr), int(intervals[1][1] * sr)
        jdx = 2
        Segs_List = []
        label_List = []
        tr = 3000
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
    neucoughset = NEUCoughReader()
    Segs_List, label_List = neucoughset.get_sample_label_list(wavname="20240921111118")
    print(len(Segs_List), '\n', label_List)
