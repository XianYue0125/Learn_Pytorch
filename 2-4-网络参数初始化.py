import torch
import torch.nn as nn


# 1. 均匀分布初始化
def test01():

    # 输入数据的特征维度是5，输出维度是3
    linear = nn.Linear(5, 3)
    nn.init.uniform_(linear.weight)
    print(linear.weight)


# 2. 固定初始化
def test02():

    linear = nn.Linear(5, 3)
    nn.init.constant_(linear.weight, 5)
    print(linear.weight)


# 3. 全0初始化
def test03():

    # 偏置默认初始化为0，但是神经网络的权重不要初始化为0
    linear = nn.Linear(5, 3)
    nn.init.zeros_(linear.weight)
    print(linear.weight)


# 4. 全1初始化
def test04():

    linear = nn.Linear(5, 3)
    nn.init.ones_(linear.weight)
    print(linear.weight)


# 5. 随机初始化(正态分布)
def test05():

    linear = nn.Linear(5, 3)
    nn.init.normal_(linear.weight, mean=0, std=1)
    print(linear.weight)


# 6. kaming 初始化
def test06():

    # 正态分布的kaming初始化
    linear = nn.Linear(5, 3)
    nn.init.kaiming_normal_(linear.weight)
    print(linear.weight)

    # 均匀分布的kaming初始化
    linear = nn.Linear(5, 3)
    nn.init.kaiming_uniform_(linear.weight)
    print(linear.weight)


# 7. xavier 初始化
def test07():

    # 正态分布的xavier初始化
    linear = nn.Linear(5, 3)
    nn.init.xavier_normal_(linear.weight)
    print(linear.weight)

    # 均匀分布的xavier初始化
    linear = nn.Linear(5, 3)
    nn.init.xavier_uniform_(linear.weight)
    print(linear.weight)


if __name__ == '__main__':
    test07()