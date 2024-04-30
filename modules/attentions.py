#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/30 10:12
# @Author: ZhaoKe
# @File : attentions.py
# @Software: PyCharm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DotAttention(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, values, query):
        attention_weights = self._get_weights(values, query)
        representations = torch.bmm(values.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        return representations

    def _get_weights(self, values, query):
        hidden = query.squeeze(0)
        attention_weights = torch.bmm(values, hidden.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(attention_weights, 1)
        return attention_weights


class SoftAttention(nn.Module):
    '''
    https://arxiv.org/abs/1803.10916
    '''

    def __init__(self, emb_dim, attn_dim):
        super().__init__()
        self.attn_dim = attn_dim
        self.emb_dim = emb_dim
        self.W = torch.nn.Linear(self.emb_dim, self.attn_dim)
        self.v = nn.Parameter(torch.Tensor(self.attn_dim), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.attn_dim)

        for weight in self.v:
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, values):
        attention_weights = self._get_weights(values)
        values = values.transpose(1, 2)
        weighted = torch.mul(values, attention_weights.unsqueeze(1).expand_as(values))
        representations = weighted.sum(2).squeeze()
        return representations

    def _get_weights(self, values):
        batch_size = values.size(0)
        weights = self.W(values)
        weights = torch.tanh(weights)
        e = weights @ self.v
        attention_weights = torch.softmax(e.squeeze(1), dim=-1)
        return attention_weights
