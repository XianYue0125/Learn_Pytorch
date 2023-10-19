import torch
import numpy as np


# 1. from_numpy 函数的用法
def test01():

    data_numpy = np.array([2, 3, 4])
    data_tensor = torch.from_numpy(data_numpy.copy())

    print(type(data_numpy))
    print(type(data_tensor))

    # 默认共享内存
    data_numpy[0] = 100
    # data_tensor[0] = 100
    print(data_numpy)
    print(data_tensor)


# 2. torch.tensor 函数的用法
def test02():

    data_numpy = np.array([2, 3, 4])
    data_tensor = torch.tensor(data_numpy)

    # 默认共享内存
    # data_numpy[0] = 100
    data_tensor[0] = 100
    print(data_numpy)
    print(data_tensor)


if __name__ == '__main__':
    test02()