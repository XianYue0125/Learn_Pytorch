import torch


# 1. 使用 mul 函数
def test01():

    data1 = torch.tensor([[1, 2], [3, 4]])
    data2 = torch.tensor([[5, 6], [7, 8]])

    data = data1.mul(data2)
    print(data)


# 2. 使用 * 号运算符
def test02():

    data1 = torch.tensor([[1, 2], [3, 4]])
    data2 = torch.tensor([[5, 6], [7, 8]])

    data = data1 * data2
    print(data)


if __name__ == '__main__':
    test01()
    test02()