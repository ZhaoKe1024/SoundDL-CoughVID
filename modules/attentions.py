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

    def get_weights(self, values, query):
        hidden = query.squeeze(0)
        attention_weights = torch.bmm(values, hidden.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(attention_weights, 1)
        return attention_weights


class SoftAttention(nn.Module):
    """
    https://arxiv.org/abs/1803.10916
    """

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

    def get_weights(self, values, query=None):
        batch_size = values.size(0)
        weights = self.W(values)
        weights = torch.tanh(weights)
        e = weights @ self.v
        attention_weights = torch.softmax(e.squeeze(1), dim=-1)
        return attention_weights


class SimpleSelfAttention(nn.Module):
    """
    一个简化的多头自注意力层，仅 1 层。
    """

    def __init__(self, d_model, n_heads):
        super(SimpleSelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: (batch, num_tokens, d_model)
        """
        B, N, D = x.size()
        # B: batch_size, N: token数(这里是32)，D: d_model

        # 线性映射
        Q = self.query_proj(x)  # (B, N, D)
        K = self.key_proj(x)  # (B, N, D)
        V = self.value_proj(x)  # (B, N, D)

        # 拆成多头
        #  (B, N, n_heads, head_dim) -> 调整顺序便于做注意力计算
        Q = Q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, N, head_dim)
        K = K.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # 注意力分数: Q*K^T / sqrt(d_k)
        # K.transpose(-2, -1) -> (B, n_heads, head_dim, N)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # (B, n_heads, N, N)

        # softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, n_heads, N, N)

        # 加权求和
        out = torch.matmul(attn_weights, V)  # (B, n_heads, N, head_dim)

        # 拼回原来的维度
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        # 输出层
        out = self.out_proj(out)  # (B, N, D)
        return out, attn_weights


class SimpleGCNLayer(nn.Module):
    """
    一个简化的 GCN 实现 (单层)。
    A is the adjacency matrix (必须事先给定或动态构造)
    """
    def __init__(self, in_channels, out_channels):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, adj):
        """
        x: (batch, num_nodes, in_channels)
        adj: (num_nodes, num_nodes) 或 (batch, num_nodes, num_nodes)
        """
        # 如果 adj 不随 batch 而变化，则可以 broadcast 到 (batch, num_nodes, num_nodes)
        if len(adj.shape) == 2:
            # adj 形状是 (num_nodes, num_nodes)
            adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)

        # x 形状: (batch, num_nodes, in_channels)
        # 先做线性变换
        out = self.linear(x)  # (batch, num_nodes, out_channels)

        # 邻接矩阵消息传播
        # 这里的实现是最基本的 A * X 形式，并没有做归一化，你可根据需要做 D^-1/2 * A * D^-1/2
        out = torch.bmm(adj, out)  # (batch, num_nodes, out_channels)

        return out
