import torch


# 1. 均值
def test01():

    torch.manual_seed(0)
    # data = torch.randint(0, 10, [2, 3], dtype=torch.float64)
    data = torch.randint(0, 10, [2, 3]).double()
    # print(data.dtype)

    print(data)
    # 默认对所有的数据计算均值
    print(data.mean())
    # 按指定的维度计算均值
    print(data.mean(dim=0))
    print(data.mean(dim=1))


# 2. 求和
def test02():

    torch.manual_seed(0)
    data = torch.randint(0, 10, [2, 3]).double()

    print(data.sum())
    print(data.sum(dim=0))
    print(data.sum(dim=1))


# 3. 平方
def test03():

    torch.manual_seed(0)
    data = torch.randint(0, 10, [2, 3]).double()
    print(data)
    data = data.pow(2)
    print(data)


# 4. 平方根
def test04():

    torch.manual_seed(0)
    data = torch.randint(0, 10, [2, 3]).double()
    print(data)
    data = data.sqrt()
    print(data)


# 5. e多少次方
def test05():

    torch.manual_seed(0)
    data = torch.randint(0, 10, [2, 3]).double()
    print(data)
    data = data.exp()
    print(data)


# 6. 对数
def test06():

    torch.manual_seed(0)
    data = torch.randint(0, 10, [2, 3]).double()
    print(data)
    data = data.log()     # 以e为底
    data = data.log2()    # 以2为底
    data = data.log10()   # 以10为底
    print(data)


if __name__ == '__main__':
    test06()