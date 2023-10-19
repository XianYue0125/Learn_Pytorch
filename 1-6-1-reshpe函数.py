import torch


def test():

    torch.manual_seed(0)
    data = torch.randint(0, 10, [4, 5])

    # 查看张量的形状
    print(data.shape, data.shape[0], data.shape[1])
    print(data.size(), data.size(0), data.size(1))

    # 修改张量的形状
    new_data = data.reshape(2, 10)
    print(new_data)

    # 注意: 转换之后的形状元素个数得等于原来张量的元素个数
    # new_data = data.reshape(1, 10)
    # print(new_data)

    # 使用-1代替省略的形状
    new_data = data.reshape(5, -1)
    print(new_data)

    new_data = data.reshape(-1, 2)
    print(new_data)


if __name__ == '__main__':
    test()