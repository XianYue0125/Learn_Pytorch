import torch


# 1. 使用 cuda 方法
def test01():

    data = torch.tensor([10, 20, 30])
    print('存储设备:', data.device)

    # 将张量移动到 GPU 设备上
    data = data.cuda()
    print('存储设备:', data.device)

    # 将张量从GPU再移动到CPU
    data = data.cpu()
    print('存储设备:', data.device)


# 2. 直接将张量创建在指定设备上
def test02():

    data = torch.tensor([10, 20, 30], device='cuda:0')
    print('存储设备:', data.device)

    # 把张量移动到cpu设备上
    data = data.cpu()
    print('存储设备:', data.device)


# 3. 使用 to 方法
def test03():

    data = torch.tensor([10, 20, 30])
    print('存储设备:', data.device)

    # 使用 to 方法移动张量到指定设备
    data = data.to('cuda:0')
    print('存储设备:', data.device)



# 4. 注意: 存储在不同设备上的张量不能够直接运算
def test04():

    data1 = torch.tensor([10, 20, 30])
    data2 = torch.tensor([10, 20, 30], device='cuda:0')

    # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
    # 下面代码会报错

    # 如果你的电脑上安装 pytorch 不是 gpu 版本的，或者电脑本身没有 gpu (nvidia)设备环境
    # 否则下面的调用 cuda 函数的代码会报错
    data1 = data1.cuda()

    data = data1 + data2
    print(data)


if __name__ == '__main__':
    test04()