import torch
import torch.nn as nn


# 1. API 基本使用
def test01():

    inputs = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]).float()
    inputs = inputs.unsqueeze(0).unsqueeze(0)
    print(inputs.shape)

    # 1. 最大池化
    # 输入数据的形状: （N, C, H, W）
    polling = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    output = polling(inputs)
    print(output.shape)

    # 2. 平均池化
    polling = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    output = polling(inputs)
    print(output.shape)


# 2. stride 步长
def test02():

    inputs = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]).float()
    inputs = inputs.unsqueeze(0).unsqueeze(0)

    # 1. 最大池化
    # 输入数据的形状: （N, C, H, W）
    polling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    output = polling(inputs)
    print(output.shape)

    # 2. 平均池化
    polling = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    output = polling(inputs)
    print(output.shape)



# 3. padding 填充
def test03():

    inputs = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]).float()
    inputs = inputs.unsqueeze(0).unsqueeze(0)

    # 1. 最大池化
    # 输入数据的形状: （N, C, H, W）
    polling = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
    output = polling(inputs)
    print(output.shape)

    # 2. 平均池化
    polling = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
    output = polling(inputs)
    print(output.shape)


# 4. 多通道池化
def test04():

    inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                           [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                           [[11, 22, 33], [44, 55, 66], [77, 88, 99]]]).float()
    inputs = inputs.unsqueeze(0)
    print(inputs.shape)


    # 1. 最大池化
    # 输入数据的形状: （N, C, H, W）
    polling = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
    output = polling(inputs)
    print(output.shape)

    # 2. 平均池化
    polling = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
    output = polling(inputs)
    print(output.shape)

    # 注意： 池化计算只会改变特征图的大小，但是不会改变输入图像的通道数量


if __name__ == '__main__':
    test04()
