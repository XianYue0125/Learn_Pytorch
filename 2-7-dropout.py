import torch
import torch.nn as nn


# 1. 创建以及使用dropout
def test01():

    # 初始化 Dropout 对象
    dropout = nn.Dropout(p=0.8)
    # 初始化数据
    inputs = torch.randint(0, 10, size=[5, 8]).float()
    print(inputs)

    print('-' * 50)

    # 将 inputs 输入经过 dropout
    outputs =  dropout(inputs)

    # 每个输入的数据会有 p 的概率设置为 0
    print(outputs)


# 2. dropout 随机丢弃对网络参数的影响
# 达到的效果：降低网络的复杂度，具有正则化的作用
def test02():

    # 固定随机数种子
    torch.manual_seed(0)

    # 初始化权重
    w = torch.randn(15, 1, requires_grad=True)
    # 初始化输入数据
    x = torch.randint(0, 10, size=[5, 15]).float()

    # 计算梯度
    y = x @ w
    y = y.sum()
    y.backward()
    print('梯度:', w.grad.reshape(1, -1).squeeze().numpy())


def test03():

    # 固定随机数种子
    torch.manual_seed(0)

    # 初始化权重
    w = torch.randn(15, 1, requires_grad=True)
    # 初始化输入数据
    x = torch.randint(0, 10, size=[5, 15]).float()

    # 初始化丢弃层
    dropout = nn.Dropout(p=0.8)
    x = dropout(x)

    # 计算梯度
    y = x @ w
    y = y.sum()
    y.backward()
    print('梯度:', w.grad.reshape(1, -1).squeeze().numpy())


if __name__ == '__main__':
    test02()
    print('-' * 50)
    test03()