import torch


# 1. 不修改原数据的计算
def test01():

    # 第一个参数: 开始值
    # 第二个参数: 结束值
    # 第三个参数: 形状
    data = torch.randint(0, 10, [2, 3])
    print(data)

    # 计算完成之后，会返回一个新的张量
    data = data.add(10)
    print(data)

    # data.sub()  # 减法
    # data.mul()  # 乘法
    # data.div()  # 除法
    # data.neg()  # 取相反数


# 2. 修改原数据的计算(inplace方式的计算)
def test02():

    data = torch.randint(0, 10, [2, 3])
    print(data)

    # 带下划线的版本的函数直接修改原数据，不需要用新的变量保存
    data.add_(10)
    print(data)

    # data.sub_()  # 减法
    # data.mul_()  # 乘法
    # data.div_()  # 除法
    # data.neg_()  # 取相反数


if __name__ == '__main__':
    test02()