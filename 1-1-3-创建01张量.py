import torch


# 1. 创建全为0的张量
def test01():

    # 1.1 创建指定形状全为0的张量
    data = torch.zeros(2, 3)
    print(data)

    # 1.2 根据其他张量的形状去创建全0张量
    data = torch.zeros_like(data)
    print(data)


# 2. 创建全为1的张量
def test02():

    # 2.1 创建指定形状全为1的张量
    data = torch.ones(2, 3)
    print(data)

    # 2.2 根据其他张量的形状去创建全1张量
    data = torch.ones_like(data)
    print(data)


# 3. 创建全为指定值的张量
def test03():

    # 3.1 创建形状为2行3列，值全部为10的张量
    data = torch.full([2, 3], 100)
    print(data)

    # 3.2 创建一个形状和data一样，但是值全部为200的张量
    data = torch.full_like(data, 200)
    print(data)


if __name__ == '__main__':
    test03()