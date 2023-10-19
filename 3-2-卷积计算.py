import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def show(img):

    # 要求输入图像的形状: (H, W, C)
    plt.imshow(img)
    plt.axis('off')  # 去掉坐标系的刻度标签
    plt.show()


# 1. 单个卷积核
def test01():

    # 读取图像: (640, 640, 4)  --> (H, W, C)
    img = plt.imread('data/彩色图片.png')
    print(img.shape)
    show(img)

    # 构建卷积核 (B, C, H, W)
    # in_channels 输入图像的通道数
    # out_channels 指的是当输入一个图像之后，产生几个特征图。也可以理解为卷积核的数量
    # kernel_size 表示卷积核大小
    # stride 步长
    # padding 填充
    conv = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)

    # 额外注意: 卷积层对输入的数据有形状要求，(BatchSize, Channel, Height, Width)
    # 将 (H, W, C) -> (C, H, W)
    img = torch.tensor(img).permute(2, 0, 1)
    print(img.shape)

    # 将(C, H, W) -> (B, C, H, W)
    new_img = img.unsqueeze(0)
    print(new_img.shape)

    # 将数据送入到卷积层进行计算
    new_img = conv(new_img)
    print(new_img.shape)

    # 将 (B, C, H, W) -> (H, W, C)
    new_img = new_img.squeeze(0).permute(1, 2, 0)
    show(new_img.detach().numpy())


# 2. 多个卷积核
def test02():

    img = plt.imread('data/彩色图片.png')
    # out_channels=3 表示有3个卷积核，输出的特征图数量也有3个
    conv = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1)

    # 额外注意: 卷积层对输入的数据有形状要求，(BatchSize, Channel, Height, Width)
    # 将 (H, W, C) -> (C, H, W)
    img = torch.tensor(img).permute(2, 0, 1)
    print(img.shape)

    new_img = img.unsqueeze(0)

    new_img = conv(new_img)
    print(new_img.shape)

    new_img = new_img.squeeze(0).permute(1, 2, 0)
    print(new_img.shape)

    show(new_img[:, :, 0].unsqueeze(-1).detach().numpy())
    show(new_img[:, :, 1].unsqueeze(-1).detach().numpy())
    show(new_img[:, :, 2].unsqueeze(-1).detach().numpy())


if __name__ == '__main__':
    test02()