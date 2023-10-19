import torch


# 1. view 函数的使用
def test01():

    data = torch.tensor([[10, 20, 30], [40, 50, 60]])
    data = data.view(3, 2)
    print(data.shape)

    # is_contiguous 函数来判断张量是否是连续内存空间(整块的内存)
    print(data.is_contiguous())


# 2. view 函数使用注意
def test02():

    # 当张量经过 transpose 或者 permute 函数之后，内存空间基本不连续
    # 此时，必须先把空间连续，才能够使用 view 函数进行张量形状操作

    data = torch.tensor([[10, 20, 30], [40, 50, 60]])
    print('是否连续:', data.is_contiguous())
    data = torch.transpose(data, 0, 1)
    print('是否连续:', data.is_contiguous())

    # 此时，在不连续内存的情况使用 view 会怎么样呢？
    data = data.contiguous().view(2, 3)
    print(data)


if __name__ == '__main__':
    test02()