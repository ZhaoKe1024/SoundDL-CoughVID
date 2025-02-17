# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2025-02-06 23:37
import os
# import json
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
import librosa
import torch
from chapter2_VADmodel import VADModel
from chapter2_SEDmodel import SEDModel
# DATA_ROOT = "F:/DATAS/NEUCOUGHDATA_COUGH/"
DATA_ROOT = "E:/DATAS-Medical/BILIBILICOUGH/"
fname = "bilicough_004"


def preprocessing_show():
    wav_test_path = os.path.join(DATA_ROOT, fname + ".wav")
    ass_test_path = os.path.join(DATA_ROOT, fname + ".ass")
    y, sr = librosa.load(wav_test_path)
    plt.figure(0)
    plt.plot(range(len(y)), y, c="#000000")
    plt.show()


def detection():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    vad_model = VADModel()
    vad_model.load_state_dict(torch.load("./runs/c2vadmodel/202502141500/vad_model_epoch30.pth"))
    vad_model.to(device)
    vad_model.eval()
    print("load VAD model.")
    sed_model = SEDModel()
    sed_model.load_state_dict(torch.load("./runs/c2sedmodel/202502161815/sed_model_epoch30.pth"))
    sed_model.to(device)
    sed_model.eval()
    print("load SED model.")

    WAVE_ROOT = "G:/DATAS-Medical/BILIBILICOUGH/"
    testwav, sr = librosa.load(WAVE_ROOT + "bilicough_009.wav")
    N = len(testwav)
    maxv = max(testwav)
    seg_list = []
    st, step, overlap = 0, 22050, 22050 // 3
    while st + step <= N:
        seg_list.append(testwav[st:st + step])
        st = st + step - overlap
    tmp = testwav[st:]
    new_tmp = np.zeros(step)
    st = (step - len(tmp)) // 2
    new_tmp[st:st + len(tmp)] = tmp
    seg_list.append(new_tmp)
    print(len(seg_list))

    seg_list = [torch.from_numpy(it) for it in seg_list]

    batch_size = 32
    x_batchs = []
    ind = 0
    while ind + batch_size < len(seg_list):
        x_batchs.append(torch.stack(seg_list[ind:ind + batch_size], dim=0))
        ind += batch_size
    x_batchs.append(torch.stack(seg_list[ind:], dim=0))
    print("batch num:{}, batch shape:{}".format(len(x_batchs), x_batchs[0].shape))

    vad_pred_list = None
    for batch_id, x_wav in enumerate(x_batchs):
        with torch.no_grad():
            y_pred = vad_model(x=x_wav.to(device).unsqueeze(1).to(torch.float32))
            if vad_pred_list is None:
                vad_pred_list = y_pred
            else:
                vad_pred_list = torch.concat((vad_pred_list, y_pred), dim=0)
    vad_pred_list = np.argmax(vad_pred_list.data.cpu().numpy(), axis=1)
    print("VAD prediction:", vad_pred_list)

    sound_events = []
    indices = []
    for i in range(len(vad_pred_list)):
        if vad_pred_list[i] > 0.5:
            indices.append(i)
            sound_events.append(seg_list[i])
    print("sound event indicex:", indices)
    # print(sound_events)

    batch_size = 32
    x_batchs = []
    ind = 0
    while ind + batch_size < len(sound_events):
        x_batchs.append(torch.stack(sound_events[ind:ind + batch_size], dim=0))
        ind += batch_size
    x_batchs.append(torch.stack(sound_events[ind:], dim=0))
    print("batch num:{}, batch shape:{}".format(len(x_batchs), x_batchs[0].shape))
    pred_list = None
    for batch_id, x_wav in enumerate(x_batchs):
        with torch.no_grad():
            y_pred = sed_model(x=x_wav.to(device).unsqueeze(1).to(torch.float32))
            if pred_list is None:
                pred_list = y_pred
            else:
                pred_list = torch.concat((pred_list, y_pred), dim=0)
    pred_list = np.argmax(pred_list.data.cpu().numpy(), axis=1)
    print(pred_list)
    result = [-1] * len(vad_pred_list)
    for ind in range(len(indices)):
        result[indices[ind]] = pred_list[ind]
    sed_label2name = {0: "breathe", 1: "clearthroat", 2: "cough", 3: "exhale", 4: "hum", 5: "inhale",
                      6: "sniff", 7: "speech", 8: "vomit", 9: "whooping", -1: "--"}
    print([sed_label2name[it] for it in result])

    result_squeezed = []
    it = result[0]
    ind = 1
    while ind < len(result):
        if result[ind] != it:
            result_squeezed.append(it)
            it = result[ind]
        ind += 1
    print(result_squeezed)
    print([sed_label2name[it] for it in result_squeezed])


if __name__ == '__main__':
    # preprocessing_show()
    detection()
