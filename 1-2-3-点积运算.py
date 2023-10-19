import torch


# 1. 使用 @ 运算符
def test01():

    # 形状为: 3行2列
    data1 = torch.tensor([[1, 2],
                          [3, 4],
                          [5, 6]])
    # 形状为: 2行2列
    data2 = torch.tensor([[5, 6],
                          [7, 8]])
    data = data1 @ data2
    print(data)


# 2. 使用 mm 函数
def test02():

    # 要求输入的张量形状都是二维
    # 形状为: 3行2列
    data1 = torch.tensor([[1, 2],
                          [3, 4],
                          [5, 6]])
    # 形状为: 2行2列
    data2 = torch.tensor([[5, 6],
                          [7, 8]])

    data = torch.mm(data1, data2)
    print(data)


# 3. 使用 bmm 函数
def test03():

    # 第一个维度: 表示批次
    # 第二个维度: 多少行
    # 第三个维度: 多少列
    data1 = torch.randn(3, 4, 5)
    data2 = torch.randn(3, 5, 8)
    
    data = torch.bmm(data1, data2)
    print(data.shape)


# 4. 使用 matmul 函数
def test04():

    # 对二维进行计算
    data1 = torch.randn(4, 5)
    data2 = torch.randn(5, 8)
    print(torch.matmul(data1, data2).shape)
    
    # 对三维进行计算
    data1 = torch.randn(3, 4, 5)
    data2 = torch.randn(3, 5, 8)
    print(torch.matmul(data1, data2).shape)

    data1 = torch.randn(3, 4, 5)
    data2 = torch.randn(5, 8)
    print(torch.matmul(data1, data2).shape)


if __name__ == '__main__':
    test04()