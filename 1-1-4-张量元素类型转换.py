import torch


# 1. type 函数进行转换
def test01():

    data = torch.full([2, 3], 10)
    print(data.dtype)

    # 注意: 返回一个新的类型转换过的张量
    data = data.type(torch.DoubleTensor)
    print(data.dtype)


# 2. 使用具体类型函数进行转换
def test02():

    data = torch.full([2, 3], 10)
    print(data.dtype)

    # 转换成 float64 类型
    data = data.double()
    print(data.dtype)

    data = data.short()   # 将张量元素转换为 int16 类型
    data = data.int()   # 将张量转换为 int32 类型
    data = data.long()  # 将张量转换为 int64 类型
    data = data.float()  # 将张量转换为 float32


if __name__ == '__main__':
    test02()