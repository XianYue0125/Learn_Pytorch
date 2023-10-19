import torch
import torch.nn as nn


# 1. RNN 输入单个词
def test01():

    # 初始化 RNN 网络层
    # input_size 输入句子的每个词的向量的维度，比如：'我' 经过了词嵌入，得到了一个 128 维的向量表示
    # hidden_size 隐藏层的大小， 隐藏层的神经元的个数，影响到最终输出数据的维度
    rnn = nn.RNN(input_size=128, hidden_size=256)


    # 初始化输入数据
    # 注意输入的数据有两个: 上一个时间步的隐藏状态、当前时间步的输入
    # inputs 的形状 (seq_len, batch_size, input_size)
    inputs = torch.randn(1, 1, 128)
    # 隐藏层的形状 (num_layers, batch_size, hidden_size)
    # 初始的隐藏状态全部为 0
    hn = torch.zeros(1, 1, 256)


    # 将数据送入到循环网络层
    # 输出: output, hn
    output, hn = rnn(inputs, hn)
    print(output.shape)
    print(hn.shape)


# 2. RNN 输入句子
def test02():
    # 初始化循环网络层
    rnn = nn.RNN(input_size=128, hidden_size=256)
    # 构造输入数据
    # inputs 的形状 (seq_len, batch_size, input_size)
    # 输入的句子长度为8，一次输入1个句子
    inputs = torch.randn(8, 1, 128)
    # 初始化隐藏状态
    # 隐藏层的形状 (num_layers, batch_size, hidden_size)
    hn = torch.zeros(1, 1, 256)

    # 数据送入到循环网络
    output, hn = rnn(inputs, hn)
    print(output.shape)
    # hn 表示的是最后一个时间步的隐藏状态
    print(hn.shape)


# 3. RNN 输入批次的数据
def test03():

    # 初始化循环网络层
    rnn = nn.RNN(input_size=128, hidden_size=256)
    # 构造输入数据
    # inputs 的形状 (seq_len, batch_size, input_size)
    # 输入的句子长度为8，一次输入1个句子
    inputs = torch.randn(8, 16, 128)
    # 初始化隐藏状态
    # 隐藏层的形状 (num_layers, batch_size, hidden_size)
    hn = torch.zeros(1, 16, 256)

    # 数据送入到循环网络
    output, hn = rnn(inputs, hn)
    print(output.shape)
    # hn 表示的是最后一个时间步的隐藏状态
    print(hn.shape)



if __name__ == '__main__':
    test03()