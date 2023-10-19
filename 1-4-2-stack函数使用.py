import torch


def test():

    torch.manual_seed(0)
    data1 = torch.randint(0, 10, [2, 3])
    data2 = torch.randint(0, 10, [2, 3])

    print(data1)
    print(data2)
    print('-' * 30)

    # 将两个张量 stack 起来，像 cat 一样指定维度
    # 1. 按照0维度进行叠加
    # new_data = torch.stack([data1, data2], dim=0)
    # print(new_data.shape)
    # print(new_data)
    # print('-' * 30)

    # # 2. 按照1维度进行叠加
    # new_data = torch.stack([data1, data2], dim=1)
    # print(new_data.shape)
    # print(new_data)
    # print('-' * 30)

    # 3. 按照2维度进行叠加
    new_data = torch.stack([data1, data2], dim=2)
    print(new_data.shape)
    print(new_data)
    print('-' * 30)


if __name__ == '__main__':
    test()