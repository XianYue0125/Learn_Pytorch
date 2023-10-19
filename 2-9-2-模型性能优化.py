import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import time
from sklearn.preprocessing import StandardScaler


# 构建数据集
def create_dataset():

    data = pd.read_csv('data/手机价格预测.csv')

    # 特征值和目标值
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    # 数据集划分
    x_train, x_valid, y_train, y_valid = \
        train_test_split(x, y, train_size=0.8, random_state=88, stratify=y)

    # 数据标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_valid = transfer.transform(x_valid)

    # 构建数据集
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train.values))
    valid_dataset = TensorDataset(torch.from_numpy(x_valid), torch.tensor(y_valid.values))

    return train_dataset, valid_dataset, x_train.shape[1], len(np.unique(y))


train_dataset, valid_dataset, input_dim, class_num = create_dataset()


# 构建网络模型
class PhonePriceModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(PhonePriceModel, self).__init__()

        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, 128)
        self.linear5 = nn.Linear(128, output_dim)

    def _activation(self, x):
        return torch.sigmoid(x)

    def forward(self, x):

        x = self._activation(self.linear1(x))
        x = self._activation(self.linear2(x))
        x = self._activation(self.linear3(x))
        x = self._activation(self.linear4(x))
        output = self.linear5(x)

        return output


# 编写训练函数
def train():

    # 固定随机数种子
    torch.manual_seed(0)

    # 初始化模型
    model = PhonePriceModel(input_dim, class_num)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化方法
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # 训练轮数
    num_epoch = 50

    for epoch_idx in range(num_epoch):

        # 初始化数据加载器
        dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
        # 训练时间
        start = time.time()
        # 计算损失
        total_loss = 0.0
        total_num = 1
        # 准确率
        correct = 0

        for x, y in dataloader:

            output = model(x)
            # 计算损失
            loss = criterion(output, y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            total_num += len(y)
            total_loss += loss.item() * len(y)

        print('epoch: %4s loss: %.2f, time: %.2fs' %
              (epoch_idx + 1, total_loss / total_num, time.time() - start))

    # 模型保存
    torch.save(model.state_dict(), 'model/phone-price-model.bin')


def test():

    # 加载模型
    model = PhonePriceModel(input_dim, class_num)
    model.load_state_dict(torch.load('model/phone-price-model.bin'))

    # 构建加载器
    dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    # 评估测试集
    correct = 0
    for x, y in dataloader:

        output = model(x)
        y_pred = torch.argmax(output, dim=1)
        correct += (y_pred == y).sum()

    print('Acc: %.5f' % (correct.item() / len(valid_dataset)))


if __name__ == '__main__':
    train()
    test()
