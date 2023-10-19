import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time


# 1. 构建数据集
def create_dataset():

    # 读取数据集
    data = pd.read_csv('data/手机价格预测.csv')
    # 将特征值和目标值拆分
    x, y = data.iloc[:, :-1], data.iloc[:, -1]

    # x 的数据类型是 float64
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    # 数据集划分
    x_train, x_valid, y_train, y_valid = \
        train_test_split(x, y, train_size=0.8, random_state=88, stratify=y)

    # 构建PyTorch数据集对象
    train_dataset = TensorDataset(torch.from_numpy(x_train.values), torch.tensor(y_train.values))
    valid_dataset = TensorDataset(torch.from_numpy(x_valid.values), torch.tensor(y_valid.values))

    # 返回数据: 训练集对象、测试集对象、特征维度、类别数量
    return train_dataset, valid_dataset, x_train.shape[1], len(np.unique(y))


train_dataset, valid_dataset, input_dim, class_num = create_dataset()


class PhonePriceModel(nn.Module):

    def __init__(self, input_dim, class_num):

        # 调用父类对象的初始化函数
        super(PhonePriceModel, self).__init__()

        # 定义网络层
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        # 输出层
        self.linear3 = nn.Linear(256, class_num)

    def _activation(self, x):
        return torch.sigmoid(x)

    def forward(self, x):

        x = self.linear1(x)
        x = self._activation(x)
        x = self.linear2(x)
        x = self._activation(x)
        output = self.linear3(x)

        return output


def train():

    # 固定随机数种子
    torch.manual_seed(0)

    # 初始化网络模型
    model = PhonePriceModel(input_dim, class_num)
    # 损失函数, 会首先对数据进行 softmax, 再进行交叉熵损失计算
    criterion = nn.CrossEntropyLoss()
    # 优化方法
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    # 训练轮数, 表示将所有的训练数据完全送入到网络中多少次
    num_epochs = 100

    for epoch_idx in range(num_epochs):

        # 初始化数据加载器
        dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
        # 训练时间
        start = time.time()
        # 计算损失
        total_loss = 0.0
        total_num = 0
        # 预测正确的样本数量
        corrent = 0

        for x, y in dataloader:

            # 将数据送入网络
            output = model(x)
            # 计算损失, 计算的是平均损失
            loss = criterion(output, y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            # 累计总样本数量
            total_num += len(y)
            # 累计总损失
            total_loss += loss.item() * len(y)

            # 累计预测正确的样本数量
            y_pred = torch.argmax(output, dim=-1)
            corrent += ((y_pred == y).sum().item())

        print('epoch: %4s loss: %.2f time: %.2fs acc: %.2f' %
              (epoch_idx + 1,
               total_loss / total_num,
               time.time() - start,
              corrent / total_num))

    # 模型保存
    torch.save(model.state_dict(), 'model/phone-price-model.pth')


def test():

    # 1. 加载模型
    model = PhonePriceModel(input_dim, class_num)
    model.load_state_dict(torch.load('model/phone-price-model.pth'))

    # 2. 构建测试集数据加载器
    datalaoder = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    # 3. 计算测试集上的准确率
    correct = 0
    for x, y in datalaoder:

        # 将数据送入网络
        output = model(x)
        # 得到预测标签
        y_pred = torch.argmax(output, dim=-1)
        correct += (y_pred == y).sum().item()

    print('acc: %.5f' % (correct / len(valid_dataset)))


if __name__ == '__main__':
    test()
