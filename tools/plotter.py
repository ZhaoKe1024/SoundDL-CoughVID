#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/18 12:37
# @Author: ZhaoKe
# @File : plotter.py
# @Software: PyCharm
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.manifold import TSNE
import torch

rgb_planning_22 = ["#e6194B",  # 0
                   "#3cb44b",
                   "#ffe119",
                   "#4363d8",
                   "#f59231",
                   "#911eb4",  # 5
                   "#42d4f4",
                   "#f032e6",
                   "#bfef45",
                   "#fabed4",
                   "#469990",  # 10
                   "#dcbeff",
                   "#9A6324",
                   "#fffac8",
                   "#800000",
                   "#aaffc3",
                   "#808000",  # 16
                   "#ffd8b1",
                   "#000075",
                   "#a9a9a9",
                   "#ffffff",
                   "#000000",  # 21
                   ]


## t-SNE
def plot_tSNE(embd, names, save_path=""):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    transformed = tsne.fit_transform(embd)
    data = {
        "dim-1": transformed[:, 0],
        "dim-2": transformed[:, 1],
        "label": names,
    }

    plt.figure()
    sns.scatterplot(
        x="dim-1",
        y="dim-2",
        hue="label",
        palette=sns.color_palette(n_colors=7),
        data=data,
        legend="full",
    )
    plt.legend(loc="best", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, format="png")
    plt.show()
    plt.close()


## Umap
def plot_umap(embd, target, save_path=""):
    import umap
    reducer = umap.UMAP()
    transformed = reducer.fit_transform(embd)
    plt.figure()
    plt.scatter(transformed[:, 0], transformed[:, 1], c=target, cmap='Spectral', s=5)
    plt.colorbar()
    plt.show()
    plt.savefig(save_path, dpi=300, format="png")
    plt.close()


## heatmap
def plot_heatmap(pred_matrix, label_vec, ticks, save_path):
    max_arg = list(pred_matrix.argmax(axis=1))
    conf_mat = metrics.confusion_matrix(max_arg, label_vec)
    df_cm = pd.DataFrame(conf_mat, index=range(conf_mat.shape[0]), columns=range(conf_mat.shape[0]))
    heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.xticks(range(len(ticks)), ticks)
    plt.yticks(range(len(ticks)), ticks)
    plt.xlabel("predict label")
    plt.ylabel("true label")
    plt.savefig(save_path, dpi=300, format="png")
    # plt.show()
    plt.close()


def trim(n: int, mode: int):
    """

    :param n:
    :param mode: 0 eye 1 lowtri  2 uppertri
    :return: matrix with 2 dim
    """
    if mode == 0:
        return np.eye(n)
    matrix = np.zeros((n, n), dtype=np.int64)
    if mode == 1:
        for i in range(n):
            for j in range(i):
                matrix[i, j] = 1
    if mode == 2:
        for i in range(n):
            for j in range(i+1, n):
                matrix[i, j] = 1
    return matrix


def calc_accuracy(pred_matrix, label_vec, save_path):
    print(pred_matrix.shape)
    # output = torch.nn.functional.softmax(pred_matrix, dim=-1)
    # output = output.data.cpu().numpy()
    output = np.argmax(pred_matrix, axis=1)
    # acc = np.mean((output == labels).astype(int))

    cfm = metrics.confusion_matrix(y_true=label_vec, y_pred=output)  # .ravel()
    N, class_num = len(label_vec), cfm.shape[0]
    diag_m  = trim(class_num, mode=0)
    # low_trim, upper_trim = trim(class_num, mode=1), trim(class_num, mode=2)
    tp_vec = cfm.diagonal()
    tp = np.sum(diag_m*cfm)
    acc = tp / N
    prec = tp_vec / cfm.sum(axis=0)
    recall = tp_vec / cfm.sum(axis=1)
    # f1_s = 2.0 * prec * recall / np.maximum(prec + recall, sys.float_info.epsilon)
    print(f"acc: {acc}")
    print("precision:", ["%.4f" % val for val in prec])
    print("recall:", ["%.4f" % val for val in recall])
    with open(save_path, 'w') as fout:
        fout.write(f"acc: {acc},\n")
        fout.write(f"precision: [{','.join(['%.4f' % val for val in prec])}],\n")
        fout.write(f"recall: [{'.'.join(['%.4f' % val for val in recall])}]")
    return acc


## AUC and pAUC


## GMM


if __name__ == '__main__':
    pred = np.random.randn(128, 6)
    y_true = np.random.randint(0, 6, size=(128,))
    calc_accuracy(pred, y_true, 'result.txt')
    # cfm = np.random.randint(0, 10, size=(3, 3))
    # print(cfm)
    # print(cfm.sum(axis=0))
    # print(cfm.sum(axis=1))
