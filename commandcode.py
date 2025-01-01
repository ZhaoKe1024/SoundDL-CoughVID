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


def ffmpeg_mp42wav():
    ROOT = "G:/DATAS-Medical/BILINOISE/"
    for item in os.listdir(ROOT):
        if item[-3:] == "mp4":
            name = item.split('.')[0]
            print("ffmpeg -i {}.mp4 -f wav -ar 44100 {}.wav".format(ROOT+name, ROOT+name))
            os.system("ffmpeg -i {}.mp4 -f wav -ar 44100 {}.wav".format(ROOT+name, ROOT+name))


def ffmpeg_slicenoise():
    ROOT = "G:/DATAS-Medical/BILINOISE/"
    for item in os.listdir(ROOT):
        if item[-3:] == "wav":
            ind1 = 0
            y, sr = librosa.load(ROOT+item)
            L = len(y)
            length = sr*3//2
            overlap = sr//3
            print("length:{}, overlap:{}, sr:{}".format(length, overlap, sr))
            st = 0
            while st+length < L:
                fname = ROOT+"{}_noise_{:03}.wav".format(item, ind1)
                soundfile.write(file=fname, data=y[st:st+length], samplerate=int(sr))
                st = st+length-overlap
                ind1 += 1


if __name__ == '__main__':
    ffmpeg_slicenoise()
    # print(sec2hms(123))
    # print(sec2hms(3832))
    # print(sec2hms(643))
