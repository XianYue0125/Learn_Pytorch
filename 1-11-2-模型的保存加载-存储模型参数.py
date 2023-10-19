import torch
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size * 2)
        self.linear2 = nn.Linear(input_size * 2,  output_size)

    def forward(self, inputs):
        inputs = self.linear1(inputs)
        output = self.linear2(inputs)
        return output


# 1. 实现模型参数存储
def test01():

    # 初始化模型参数
    model = Model(128, 10)
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # 定义要存储的模型参数
    save_params = {
        'init_params': {'input_size': 128, 'output_size': 10},
        'acc_score': 0.98,
        'avg_loss': 0.86,
        'iter_num': 100,
        'optim_params': optimizer.state_dict(),
        'model_params': model.state_dict()
    }

    # 存储模型参数
    torch.save(save_params, 'model/model_params.pth')


# 2. 实现模型参数加载
def test02():

    # 从磁盘中将参数加载到内存中
    model_params = torch.load('model/model_params.pth')
    # 使用参数初始化模型
    model = Model(model_params['init_params']['input_size'], model_params['init_params']['output_size'])
    model.load_state_dict(model_params['model_params'])
    # 使用参数初始化优化器
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(model_params['optim_params'])

    # 可以加载其他参数
    print('迭代次数:', model_params['iter_num'])
    print('准确率:', model_params['acc_score'])
    print('平均损失:', model_params['avg_loss'])


if __name__ == '__main__':
    test02()