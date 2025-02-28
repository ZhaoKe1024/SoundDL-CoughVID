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


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k=64):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        # 转置最内层两个维度，其他维度广播
        scores = torch.matmul(Q, K.transpose(-1, -2)) / \
                 np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        # Fills elements of self tensor with value where mask is True.
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度(v)做softmax
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        # context: [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, V)
        # context：[[z1,z2,...],[...]]向量, attn注意力稀疏矩阵（用于可视化的）
        return context, attn


class MultiHeadAttention(nn.Module):
    """这个Attention类可以实现:
    Encoder的Self-Attention
    Decoder的Masked Self-Attention
    Encoder-Decoder的Attention
    输入：seq_len x d_model
    输出：seq_len x d_model
    """

    def __init__(self, embed_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = 64
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.W_Q = nn.Linear(embed_dim, self.d_k * n_heads,
                             bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(embed_dim, self.d_k * n_heads, bias=False)
        self.W_V = nn.Linear(embed_dim, self.d_v * n_heads, bias=False)
        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        self.fc = nn.Linear(n_heads * self.d_v, embed_dim, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
        #           线性变换               拆成多头

        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1,
                                   self.n_heads, self.d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(input_K).view(batch_size, -1,
                                   self.n_heads, self.d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1,
                                   self.n_heads, self.d_v).transpose(1, 2)

        # 因为是多头，所以mask矩阵要扩充成4维的
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        print("MyMHA attn mask:", attn_mask.shape)
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # 下面将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(
            batch_size, -1, self.n_heads * self.d_v)

        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.embed_dim).to(output.device)(output + residual), attn


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


if __name__ == '__main__':
    x_input = torch.rand(size=(32, 9, 512))
    smha = SimpleSelfAttention(d_model=512, n_heads=4)
    print(smha(x_input))
