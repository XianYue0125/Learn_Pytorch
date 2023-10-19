import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
import torch.optim as optim
from torch.utils.data import DataLoader
import time


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class ImageClassificaiton(nn.Module):

    def __init__(self):

        # 调用父类初始化函数
        super(ImageClassificaiton, self).__init__()

        # 定义卷积池化
        self.conv1 = nn.Conv2d(3, 6, stride=1, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, stride=1, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义线性层
        self.linear1 = nn.Linear(576, 120)
        self.linear2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        # 在该案例中使用的是 relu 激活函数

        # 卷积池化计算
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # 将特征图送入到全连接层，此时需要进行维度的变化
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        return self.out(x)


# 编写训练函数
def train():

    # 加载数据集
    cifar10 = CIFAR10(root='data', train=True, transform=Compose([ToTensor()]))
    # 初始化网络
    model = ImageClassificaiton()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化方法
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 训练轮数
    epochs = 100

    for epoch_idx in range(epochs):

        # 构建数据加载器
        dataloader = DataLoader(cifar10, batch_size=32, shuffle=True)
        # 样本数量
        sam_num = 0
        # 损失总和
        total_loss = 0.0
        # 开始时间
        start = time.time()
        # 正确样本数量
        correct = 0

        for x, y in dataloader:

            # 送入模型
            output = model(x)
            # 计算损失
            loss = criterion(output, y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            # 统计信息
            correct += (torch.argmax(output, dim=-1) == y).sum()
            total_loss += loss.item() * len(y)
            sam_num += len(y)

        print('epoch: %2s loss:%.5f acc:%.2f time:%.2fs' %
              (epoch_idx + 1,
               total_loss / sam_num,
               correct / sam_num,
               time.time() - start))

    # 模型的保存
    torch.save(model.state_dict(), 'model/image_classification.pth')


def test():

    # 加载测试集数据
    cifar10 = CIFAR10(root='data', train=False, download=True, transform=Compose([ToTensor()]))
    # 构建数据加载器
    dataloader = DataLoader(cifar10, batch_size=32, shuffle=False)
    # 加载模型
    model = ImageClassificaiton()
    model.load_state_dict(torch.load('model/image_classification.pth'))
    # 模型有两种状态： 训练状态(模式)、预测状态(模式)
    model.eval()

    total_correct = 0
    total_samples = 0

    for x, y in dataloader:

        # 送入网络
        output = model(x)
        # 统计预测正确的样本数量
        total_correct += (torch.argmax(output, dim=-1) == y).sum()
        total_samples += len(y)

    # 打印测试集的准确率
    print('acc: %.2f' % (total_correct / total_samples))




if __name__ == '__main__':
    test()







