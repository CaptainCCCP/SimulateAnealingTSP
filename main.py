from defs import metropolis, path_distance,\
                 gen_new_path, build_distance
from draw import draw_evolution, draw_result
import random

# SA for TSP gsy20074411
# parameter

T0 = 1000
Tend = 1e-3
epoch = 200                                                   # epoch
q = 0.9                                                       # rate


def simulated_annealing():
    t = T0                                                    # ①初始化温度
    distance = build_distance()                               # 距离
    city_cnt = len(distance)
    # ②产生初始随机解
    path = random.sample(range(0, city_cnt), city_cnt)        # 打乱内容, 长度

    draw_result(path, "init_path")                            # 初始路径

    total_distance = path_distance(path, distance)            # 计算初始距离长度

    print("inital path：", [p + 1 for p in path])
    print("inital distance：", total_distance)

    # 进化过程，每一次迭代的路径总距离
    evolution = []                                            # 空列表

    # 开始循环，采用零度法
    while t > Tend:
        for _ in range(epoch):

            new_path = gen_new_path(path)

            path, total_distance = metropolis(path, new_path, distance, t)

            evolution.append(total_distance)

        t = t * q                                             # 等比例下降

    print("EndTemperature：", t)
    print("shortest Path：", [p + 1 for p in path])
    print("shortest distance：", total_distance)

    draw_result(path, "tsp_sa_best")
    draw_evolution(evolution)


if __name__ == "__main__":
    simulated_annealing()
