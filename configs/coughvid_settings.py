# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-01-24 18:10

cough_dataset = {
    "run_save_dir": "./runs/tdnn_coughvid/",
    "model": {
        "num_class": 3,
        "input_length": 94,
        "wav_length": 48000,
        "input_dim": 512,
        "n_mels": 128,
    },
    "fit": {
        "batch_size": 64,
        "epochs": 23,
        "start_scheduler_epoch": 6
    },
}
