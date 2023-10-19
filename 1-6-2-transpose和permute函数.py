import torch


# 1. transpose 函数
def test01():

    torch.manual_seed(0)
    data = torch.randint(0, 10, [3, 4, 5])

    new_data = data.reshape(4, 3, 5)
    print(new_data.shape)

    # 直接交换两个维度的值
    new_data = torch.transpose(data, 1, 2)
    print(new_data.shape)

    # 缺点: 一次只能交换两个维度
    # 把数据的形状变成 (4, 5, 3)
    # 进行第一次交换: (4, 3, 5)
    # 进行第二次交换: (4, 5, 3)
    new_data = torch.transpose(data, 0, 1)
    new_data = torch.transpose(new_data, 1, 2)
    print(new_data.shape)


# 2. permute 函数
def test02():

    torch.manual_seed(0)
    data = torch.randint(0, 10, [3, 4, 5])

    # permute 函数可以一次性交换多个维度
    new_data = torch.permute(data, [1, 2, 0])
    print(new_data.shape)


if __name__ == '__main__':
    test02()