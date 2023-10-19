import torch


# 1. 张量转换为numpy数组
def test01():

    data_tensor = torch.tensor([2, 3, 4])
    # 将张量转换为numpy数组
    data_numpy = data_tensor.numpy()

    print(type(data_tensor))
    print(type(data_numpy))

    print(data_tensor)
    print(data_numpy)


# 2. 张量和numpy数组共享内存
def test02():

    data_tensor = torch.tensor([2, 3, 4])
    data_numpy = data_tensor.numpy()

    # 修改张量元素的值，看看numpy数组是否会发生变化？会发生变化
    # data_tensor[0] = 100
    # print(data_tensor)
    # print(data_numpy)

    # 修改numpy数组元素的值，看看张量是否会发生变化？会发生变化
    data_numpy[0] = 100
    print(data_tensor)
    print(data_numpy)


# 3. 使用copy函数实现不共享内存
def test03():

    data_tensor = torch.tensor([2, 3, 4])
    # 此处, 发生了类型转换，可以使用拷贝函数产生新的数据，避免共享内存
    data_numpy = data_tensor.numpy().copy()

    # 修改张量元素的值，看看numpy数组是否会发生变化？没有发生变化
    # data_tensor[0] = 100
    # print(data_tensor)
    # print(data_numpy)

    # 修改numpy数组元素的值，看看张量是否会发生变化？没有发生变化
    data_numpy[0] = 100
    print(data_tensor)
    print(data_numpy)


if __name__ == '__main__':
    test03()