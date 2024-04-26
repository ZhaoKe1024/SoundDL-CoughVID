#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/26 14:50
# @Author: ZhaoKe
# @File : utils.py
# @Software: PyCharm
import os
import random
import numpy as np
import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        # if any(m.bias):
        torch.nn.init.constant_(m.bias, 0.)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1.)
        torch.nn.init.constant_(m.bias, 0.)


def load_ckpt(model, resume_model, m_type=None, load_epoch=None):
    if m_type is None:
        state_dict = torch.load(os.path.join(resume_model, f'model_{load_epoch}.pth'))
    else:
        if load_epoch:
            state_dict = torch.load(os.path.join(resume_model, f'model_{m_type}_{load_epoch}.pth'))
        else:
            state_dict = torch.load(os.path.join(resume_model, f'model_{m_type}.pth'))
    model.load_state_dict(state_dict)
