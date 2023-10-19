import torch
import torch.nn as nn



if __name__ == '__main__':

    # 输入的形状: [batch_size, channel, height, width]
    inputs = torch.randint(0, 10, [2, 2, 3, 3]).float()
    print(inputs)
    print('-' * 50)

    # num_features 表示每个样本特征图的数量，通道数量
    # affine 为 False 表示不带 gama 和 beta 两个学习参数
    # eps 小常数，避免分母为 0
    bn = nn.BatchNorm2d(num_features=2, affine=False, eps=1e-5)
    result = bn(inputs)
    print(result)

    # 均值是每个样本对应通道的均值
    # 方差对应通道的方差


