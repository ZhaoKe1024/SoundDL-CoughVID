#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/30 10:06
# @Author: ZhaoKe
# @File : loss.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=-1)  # prob pred
        # print("pred:", P)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        # print(ids)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        # print("alpha:", alpha)
        probs = (P * class_mask).sum(1).view(-1, 1)
        # print("probs:")
        # print(probs)
        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


if __name__ == '__main__':
    num_classes = 3
    # 定义自定义权重
    # 这里只是示例，你可以根据你的需求自行设置权重
    # 这里使用了相同的权重，你可以根据类别的重要性来设置不同的权重
    class_weights = torch.tensor([1.0, 2.0, 3.0])
    # 创建CrossEntropyLoss损失函数，并传入自定义权重
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # 假设你有一个预测的张量和一个目标张量
    # 假设批次大小为4
    # 这里的预测张量的形状是(4, num_classes)，目标张量的形状是(4,)
    # 其中预测张量的每一行是对应样本的类别预测概率，目标张量是真实的类别标签
    # 你需要根据你的具体情况来替换这些示例数据
    predictions = torch.tensor([[0.2, 0.5, 0.3],
                                [0.1, 0.2, 0.7],
                                [0.8, 0.1, 0.1],
                                [0.4, 0.4, 0.2]])
    targets = torch.tensor([0, 1, 2, 0])
    # 计算损失
    loss = criterion(predictions, targets)
    print(loss)
