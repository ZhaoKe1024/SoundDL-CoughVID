#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/11/14 22:02
# @Author: ZhaoKe
# @File : neucough_reader.py
# @Software: PyCharm
# import sys
# sys.path.append(r'C:/Program Files (zk)/PythonFiles/AClassification/SoundDL-CoughVID')
import pandas as pd
import librosa
ROOT = "F:/DATAS/NEUCOUGHDATA_COUGHSINGLE/"


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
    st_cur, en_cur = int(intervals[0][0]*sr), int(intervals[0][1]*sr)
    st_tail, en_tail = int(intervals[1][0]*sr), int(intervals[1][1]*sr)
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
                    if en_pos - st_tail>tr:
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
        Segs_List.append(sample[st_pos:en_pos+overlap])
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
    Segs_List, label_List = read_audio_for_silence(wavname="20240921111118")
    print(len(Segs_List), '\n', label_List)
