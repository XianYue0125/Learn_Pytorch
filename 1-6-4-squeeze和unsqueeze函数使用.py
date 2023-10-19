import torch


# 1. squeeze 函数使用
def test01():

    data = torch.randint(0, 10, [1, 3, 1, 5])
    print(data.shape)

    # 维度压缩, 默认去掉所有的1的维度
    new_data = data.squeeze()
    print(new_data.shape)

    # 指定去掉某个1的维度
    new_data = data.squeeze(2)
    print(new_data.shape)


# 2. unsqueeze 函数使用
def test02():

    data = torch.randint(0, 10, [3, 5])
    print(data.shape)

    # 可以在指定位置增加维度
    # -1 代表最后一个维度
    new_data = data.unsqueeze(-1)
    print(new_data.shape)


if __name__ == '__main__':
    test02()