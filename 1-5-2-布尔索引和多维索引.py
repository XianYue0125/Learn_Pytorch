import torch


# 1. 布尔索引
def test01():

    torch.manual_seed(0)
    data = torch.randint(0, 10, [4, 5])
    print(data)

    # 希望能够获得该张量中所有大于3的元素
    print(data[data > 3])

    # 希望返回第2列元素大于6的行
    print(data[data[:, 1] > 6])

    # 希望返回第2行元素大于3的所有列
    print(data[:, data[1] > 3])


# 2. 多维索引
def test02():

    torch.manual_seed(0)
    data = torch.randint(0, 10, [3, 4, 5])
    print(data)
    print('-' * 30)

    # 按照第0个维度选择第0元素，4行5列元素
    print(data[0, :, :])
    print('-' * 30)

    # 按照第1个维度选择第0元素
    print(data[:, 0, :])
    print('-' * 30)

    # 按照第2个维度选择第0元素
    print(data[:, :, 0])
    print('-' * 30)


if __name__ == '__main__':
    test02()