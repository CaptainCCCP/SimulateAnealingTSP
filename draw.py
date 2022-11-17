from matplotlib import pyplot as plt
from defs import X


# 绘制结果
def draw_result(best, file_name="tsp_sa"):
    # 各个城市的横纵坐标
    x = [pos[0] for pos in X]
    y = [pos[1] for pos in X]
    # 绘图中文设置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    # 清空画布
    plt.clf()
    # 绘制箭头
    for i in range(len(X)):
        # 箭头开始坐标
        start = X[best[i]]
        # 箭头结束坐标
        end = X[best[i + 1]] if i < len(best) - 1 else X[best[0]]
        plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                  head_width=0.2, lw=1, length_includes_head=True)
    # 绘制城市编号
    for i in range(len(X)):
        plt.text(x[best[i]], y[best[i]], "{}".format((best[i] + 1)), size=15, color="r")
    plt.xlabel(u"横坐标")
    plt.ylabel(u"纵坐标")
    plt.savefig(file_name + ".png", dpi=800)
    plt.show()
# 绘制进化过程


def draw_evolution(evolution):
    x = [i for i in range(len(evolution))]
    # 清空画布
    plt.clf()
    plt.plot(x, evolution)
    plt.savefig('tsp_sa_evolution.png', dpi=800)
    plt.show()
