import torch
import numpy as np


# 1. 根据已有的数据创建张量
def test01():

    # 1.1 创建标量
    data = torch.tensor(10)
    print(data)

    # 1.2 使用numpy数组来创建张量
    data = np.random.randn(2, 3)
    data = torch.tensor(data)
    print(data)

    # 1.3 使用list列表创建张量
    data = [[10., 20., 30.], [40., 50., 60.]]
    data = torch.tensor(data)
    print(data)


# 2. 创建指定形状的张量
def test02():

    # 2.1 创建2行3列的张量
    data = torch.Tensor(2, 3)
    print(data)

    # 2.2 可以创建指定值的张量
    # 注意: 传递列表
    data = torch.Tensor([2, 3])
    print(data)

    data = torch.Tensor([10])
    print(data)


# 3. 创建指定类型的张量
def test03():

    # 前面创建的张量都是使用默认类型或者元素类型
    # 创建一个 int32 类型的张量
    data = torch.IntTensor(2, 3)
    print(data)

    # torch.ShortTensor(2, 3)  # 表示创建的是 int16 张量
    # torch.LongTensor(2, 3)  # 表示创建的是 int32 张量
    # torch.FloatTensor(2, 3)  # 表示创建的是 float32 张量


    # 注意: 如果创建指定类型的张量，但是传递的数据不匹配，会发生类型转换
    data = torch.IntTensor([2.5, 3.5])
    print(data)


if __name__ == '__main__':
    test03()