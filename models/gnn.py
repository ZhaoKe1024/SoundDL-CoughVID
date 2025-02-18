#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/2/18 15:45
# @Author: ZhaoKe
# @File : gnn.py
# @Software: PyCharm
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def get_data_item(node_num, node_dim, edge_num):
    features = torch.rand(size=(node_num, node_dim))
    edge_index = [[], []]
    for i in range(edge_num):
        st = random.randint(0, node_num - 1)
        en = random.randint(0, node_num - 1)
        edge_index[0].append(st)
        edge_index[1].append(en)
        edge_index[0].append(en)
        edge_index[1].append(st)
    edge_index = torch.from_numpy(np.array(edge_index))
    edge_index = to_undirected(edge_index, num_nodes=node_num)
    return features, edge_index


if __name__ == '__main__':
    # ======build data==============
    # data = Planetoid(root='/data/CiteSeer', name='CiteSeer')
    class_num = 5
    # cough, wheeze, inhalation, exhalation, vomit, whooping; sniff; clearthroat, hum, speech
    event_num = 6
    batch_size = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    features, edge_index = get_data_item(node_num=64, node_dim=16, edge_num=96)
    graph = Data(x=features.to(device), edge_index=edge_index.to(device))
    print(graph)
    print(features.size(0), features.size(1))
    # 示例边索引 (无向图边: 0-1, 1-2, 2-3, 3-0)
    print(edge_index)

    gcn_model = GCN(in_channels=graph.x.shape[1], hidden_channels=32, out_channels=class_num).to(device)
    print(gcn_model)
    # 创建图数据对象

    graph = graph
    pred = gcn_model(x=graph.x, edge_index=graph.edge_index)
    print(pred.argmax(dim=-1))
