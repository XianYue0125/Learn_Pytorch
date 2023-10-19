import torch


# 1. 简单行列索引
def test01():

    # 固定随机数种子
    torch.manual_seed(0)

    data = torch.randint(0, 10, [4, 5])
    print(data)
    print('-' * 30)

    # 1.1 获得指定的某行元素
    print(data[2])

    # 1.2 获得指定的某个列的元素
    # 逗号前面表示行, 逗号后面表示列

    # 冒号表示所有行或者所有列
    # print(data[:, :])

    # 表示获得第一列的元素
    print(data[:, 2])

    # 获得指定位置的某个元素
    print(data[1, 2], data[1][2])

    # 表示先获得前三行，然后再获得第三列的数据
    print(data[:3, 2])

    # 表示获得前三行的前两列
    print(data[:3, :2])


# 2. 列表索引
def test02():


    # 固定随机数种子
    torch.manual_seed(0)

    data = torch.randint(0, 10, [4, 5])
    print(data)
    print('-' * 30)

    # 如果索引的行列都是一个1维的列表，那么两个列表的长度必须相等
    # 表示获得 (0, 0)、(2, 1)、(3, 2) 三个位置的元素
    # print(data[[0, 2, 3], [0, 1, 2]])

    # 表示获得 0、2、3 行的 0、1、2 列
    print(data[[[0], [2], [3]], [0, 1, 2]])


if __name__ == '__main__':
    test02()