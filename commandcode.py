#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/1/1 16:59
# @Author: ZhaoKe
# @File : commandcode.py
# @Software: PyCharm
import os

import librosa
import soundfile


def sec2hms(sec: int) -> str:
    h, m = divmod(sec, 3600)
    m, s = divmod(m, 60)
    return "{:02}:{:02}:{:02}".format(h, m, s)


def ffmpeg_slicenoise():
    ROOT = "G:/DATAS-Medical/BILINOISE/"
    for item in os.listdir(ROOT):
        if item[-3:] == "wav":
            ind1 = 0
            y, sr = librosa.load(ROOT + item)
            L = len(y)
            length = sr * 3 // 2
            overlap = sr // 3
            print("length:{}, overlap:{}, sr:{}".format(length, overlap, sr))
            st = 0
            while st + length < L:
                fname = ROOT + "{}_noise_{:03}.wav".format(item, ind1)
                soundfile.write(file=fname, data=y[st:st + length], samplerate=int(sr))
                st = st + length - overlap
                ind1 += 1


def add_new_instance_to_bilicough(fname):
    """在BiliCough中添加新的数据
    给定一条长音频，我手动在Aegisub软件中标注好ass文件，然后通过这个函数去简易地切分为目标长度的短片段"""
    print("file name:", fname)
    label_dict = dict()
    label_names = ["breathe", "cough", "clearthroat", "exhale", "hum", "inhale", "noise", "silence", "sniff", "speech",
                   "vomit", "whooping"]
    label_cnt = dict()
    name2label = {"breathe": 0, "cough": 2, "clearthroat": 1, "exhale": 3, "hum": 4, "inhale": 5, "noise": 6,
                  "silence": 7, "sniff": 8, "speech": 9, "vomit": 10, "whooping": 11}
    metainfo_file = open("G:/DATAS-Medical/BILIBILICOUGH/new_file_{}.csv".format(fname), 'w')
    metainfo_file.write("filename,st,en,labelfull,labelname,label\n")
    wavtest = fname + ".wav"
    asstest = fname + ".ass"
    assfin = open("G:/DATAS-Medical/BILIBILICOUGH/" + asstest, 'r', encoding="utf-8")
    label_list = []
    line = assfin.readline()
    while line.strip() != "[Events]":
        line = assfin.readline()
        # print(line)
    assfin.readline()
    line = assfin.readline()
    while line:
        # print(line)
        parts = line.split(',')
        lab_tmp = parts[9].strip()
        if lab_tmp == "useless":
            pass
        # if lab_tmp == "clearingthroat":
        #     print(name_list[idx])
        else:
            label_list.append([parts[1], parts[2], lab_tmp])
            if lab_tmp not in label_dict:
                label_dict[lab_tmp] = 1
            else:
                label_dict[lab_tmp] = label_dict.get(lab_tmp) + 1
            label = None
            if lab_tmp[:3] == "hum":
                label = lab_tmp[:3]
            elif lab_tmp[:5] in ["cough", "noise", "sniff", "vomit"]:
                label = lab_tmp[:5]
            elif lab_tmp[:6] in ["inhale", "exhale", "speech"]:
                label = lab_tmp[:6]
            elif lab_tmp[:7] in ["breathe", "silence"]:
                label = lab_tmp[:7]
            elif lab_tmp[:8] in ["whooping"]:
                label = lab_tmp[:8]
            elif lab_tmp[:11] in ["clearthroat"]:
                label = lab_tmp[:11]
            else:
                print(lab_tmp)
                raise Exception("Unknown Class.")
            if label not in label_cnt:
                label_cnt[label] = 1
            else:
                label_cnt[label] = label_cnt.get(label) + 1
            metainfo_file.write(
                "{},{},{},{},{},{}\n".format(fname, parts[1], parts[2], lab_tmp, label, name2label[label]))
            print("{},{},{},{},{},{}\n".format(fname, parts[1], parts[2], lab_tmp, label, name2label[label]))
        line = assfin.readline()
    metainfo_file.close()
    # for item in label_list:
    #     print(item)
    print("标签分布：")
    for k, v in label_dict.items():
        print("key:{},\tcount:{}".format(k, v))
    print("---------------=============----------------")
    for k, v in label_cnt.items():
        print("key:{},\tcount:{}".format(k, v))
    print("运行完毕，记得把新表格里面的数据，手动追加到bilicough_metainfo.csv里面")


if __name__ == '__main__':
    ffmpeg_slicenoise()
    # print(sec2hms(123))
    # print(sec2hms(3832))
    # print(sec2hms(643))
