#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/12/4 15:54
# @Author: ZhaoKe
# @File : test_code.py
# @Software: PyCharm
"""
暂时测试代码用
"""


def redundancy_overlap_generate():
    length = 46
    data_length = 11
    cnt_sum = length // data_length + 1
    res = cnt_sum * data_length - length
    print(cnt_sum, res)
    overlap = res // (cnt_sum-1)
    print(overlap)
    st = 0
    while st+data_length <= length:
        print("[{}, {}]".format(st, st+data_length))
        st += data_length-overlap
    print("[{}, {}]".format(length-data_length-1, length-1))


def min2sec(t: str):
    parts = t.split(':')
    res = float(parts[-1])
    f = 60
    for i in range(len(parts)-1):
        res += int(parts[len(parts)-2-i]) * f
        f *= 60
    return res


if __name__ == '__main__':
    # redundancy_overlap_generate()
    print(min2sec("00:01:19.06")*22050)
