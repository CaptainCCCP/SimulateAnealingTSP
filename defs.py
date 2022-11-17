# Metropolis准则函数
import copy
import math
import random
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.common.initializer import Zero
X = [(2374, 197),
     (2166, 878),
     (644, 5),
     (268, 3149),
     (706, 2256),
     (2180, 2557),
     (1397, 771),
     (1356, 42),
     (430, 1110),
     (2827, 1530),
     (762, 2418),
     (1150, 1031),
     (2227, 1932),
     (690, 2022)]


X_np = np.array(X)
x_tensor = Tensor(X_np)       # 14, 2


# 构建距离矩阵
def build_distance():
    # 初始化城市距离矩阵
    distance = [[0 for _ in range(len(X))] for _ in range(len(X))]
    #distance = mindspore.Tensor(shape=(x_tensor.size/2, x_tensor.size/2), dtype=mindspore.int32, init=Zero())
    # 全0
    # 计算各个城市之间的距离
    for i in range(len(X)):
        pos1 = X[i]
        for j in range(i+1, len(X)):
            pos2 = X[j]
            distance[i][j] = pow((pow(pos1[0] - pos2[0], 2) + pow(pos1[1] - pos2[1], 2)), 0.5)  # 求距离
            distance[j][i] = distance[i][j]
    return distance


# 产生新的路径解
def gen_new_path(path):
    new_path = copy.copy(path)
    idx1 = random.randint(0, len(path) - 1)   # 0到len-1之间的随机数
    idx2 = random.randint(0, len(path) - 1)
    # 交换路径中的两个城市
    temp = new_path[idx1]
    new_path[idx1] = new_path[idx2]
    new_path[idx2] = temp
    return new_path


# 计算路径总距离
def path_distance(path, distance):
    sum_distance = 0.0
    for i in range(len(path)):
        if i == len(path) - 1:
            sum_distance += distance[path[i]][path[0]]
        else:
            sum_distance += distance[path[i]][path[i + 1]]
    return sum_distance


def metropolis(old_path, new_path, distance, t):
    # 上一条路径 下一条路径 上一条距离 温度

    # E(n+1)-E(n)
    differ = path_distance(new_path, distance) - path_distance(old_path, distance)
    # E(n+1)<E(n)        p = 1
    if differ < 0:
        return copy.copy(new_path), path_distance(new_path, distance)

    # E(n+1) >= E(n)     p = exp(-E(n+1)<E(n)/t)
    if math.exp(-differ/t) >= random.uniform(0, 1):
        return copy.copy(new_path), path_distance(new_path, distance)
    # else
    return copy.copy(old_path), path_distance(old_path, distance)

