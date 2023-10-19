import torch
import matplotlib.pyplot as plt


# 1. 没有指数加权平均
def test01():

    # 固定随机数种子
    torch.manual_seed(0)

    # 随机产生30天的温度
    temperature = torch.randn(size=[30, ]) * 10

    # 绘制平均温度值
    days = torch.arange(1, 31, 1)
    plt.plot(days, temperature, 'o-r')
    plt.show()


# 2. 有指数加权平均
def test02(beta=0.9):

    # 固定随机数种子
    torch.manual_seed(0)
    # 随机产生30天的温度
    temperature = torch.randn(size=[30, ]) * 10
    days = torch.arange(1, 31, 1)

    # 存储历史指数加权平均值
    exp_weight_avg = []

    for idx, temp in enumerate(temperature, 1):

        if idx == 1:
            exp_weight_avg.append(temp)
            continue

        new_temp = exp_weight_avg[idx - 2] * beta + (1 - beta) * temp
        exp_weight_avg.append(new_temp)

    # 绘制指数加权平均温度值
    days = torch.arange(1, 31, 1)
    plt.plot(days, exp_weight_avg, 'o-r')
    plt.show()


if __name__ == '__main__':
    test02(0.5)
    test02(0.9)