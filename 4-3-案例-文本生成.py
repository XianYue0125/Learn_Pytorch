import torch
import re
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


# 构建词典
def build_vocab():

    fname = 'data/jaychou_lyrics.txt'

    # 1. 文本数据清洗
    clean_sentences = []
    for line in open(fname, 'r'):

        # 去除指定的一些内容
        line = line.replace('韩语Rap译文〗', '')
        # 保留中文、数字、部分的标点符号
        line = re.sub(r'[^\u4e00-\u9fa5 a-zA-Z0-9!?,]', '', line)
        # 连续空格替换
        line = re.sub(r'[ ]{2,}', '', line)
        # 去除行两侧的空白和换行符
        line = line.strip()
        # 去除单字的行
        if len(line) <= 1:
            continue

        if line not in clean_sentences:
            clean_sentences.append(line)

    # print(clean_sentences)

    # 2. 分词
    all_sentences = []
    index_to_word = []   # 重要的词表: 索引到词的映射
    for line in clean_sentences:
        words = jieba.lcut(line)
        # 便于后面我们将句子转换为索引表示
        all_sentences.append(words)

        for word in words:
            if word not in index_to_word:
                index_to_word.append(word)

    # 重要词表: 词到索引的映射
    word_to_index = {word: idx for idx, word in enumerate(index_to_word)}

    # 3. 将输入的语料转换为索引表示
    corpus_index = []   # 语料的索引表示
    for sentence in all_sentences:
        temp = []
        for word in sentence:
            temp.append(word_to_index[word])

        # 在每行歌词的后面添加一个空格
        temp.append(word_to_index[' '])
        corpus_index.extend(temp)

    # print(corpus_index)
    return index_to_word, word_to_index, len(index_to_word), corpus_index


# 调用构建词典的函数
index_to_word, word_to_index, word_len, corpus_index = build_vocab()


# 编写数据集类
class LyricsDataset:

    def __init__(self, corpus_index, num_chars):
        """
        :param corpus_index: 语料的索引表示
        """

        # 语料数据
        self.corpus_index = corpus_index
        # 语料长度
        self.num_chars = num_chars
        # 语料词的总数量
        self.word_count = len(corpus_index)
        # 计算句子长度有多少个
        self.number = self.word_count //  self.num_chars
    
    def __len__(self):
        return self.number
    
    def __getitem__(self, idx):
        # 输入的 idx 可能不是合法的
        start = min(max(idx, 0), self.word_count - self.num_chars - 2)
        # 获取一条样本，就会有 x, 就会有 y
        x = self.corpus_index[start: start + self.num_chars]
        y = self.corpus_index[start + 1: start + 1 + self.num_chars]

        # x = 0, 1, 2, 39, 0
        # y = 1, 2, 39, 0, 3
        
        return torch.tensor(x), torch.tensor(y)


def test01():

    index_to_word, word_to_index, word_len, corpus_index = build_vocab()
    lyrics = LyricsDataset(corpus_index, 5)
    # 注意: batch_size = 1
    dataloader = DataLoader(lyrics, shuffle=False, batch_size=1)

    for x, y in dataloader:
        print(x)
        print(y)
        break


# 构建循环神经网络
class TextGenerator(nn.Module):

    def __init__(self):
        
        super(TextGenerator, self).__init__()

        # 初始化词嵌入层
        self.ebd = nn.Embedding(num_embeddings=word_len, embedding_dim=128)
        # 初始化循环网络层
        self.rnn = nn.RNN(input_size=128, hidden_size=128)
        # 初始化输出层, 预测的标签数量为词典中词的总数量
        self.out = nn.Linear(128, word_len)

    def forward(self, inputs, hidden):

        # embed 的形状是 (1, 5, 128)
        embed = self.ebd(inputs)

        # 正则化
        embed = F.dropout(embed, p=0.2)

        # 送入循环网络层
        # output 表示的是每一个时间步的输出
        output, hidden = self.rnn(embed.transpose(0, 1), hidden)

        # 将 output 送入到全连接层得到输出
        output = self.out(output)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, 128)


def test02():

    index_to_word, word_to_index, word_len, corpus_index = build_vocab()
    lyrics = LyricsDataset(corpus_index, 5)
    # 注意: batch_size = 1
    dataloader = DataLoader(lyrics, shuffle=False, batch_size=1)

    # 初始化网络对象
    model = TextGenerator()

    for x, y in dataloader:
        # 初始化隐藏状态
        hidden = model.init_hidden()
        y_pred, hidden = model(x, hidden)
        print(y_pred.shape)
        break


# 训练函数
def train():

    # 构建词典
    index_to_word, word_to_index, word_len, corpus_index = build_vocab()
    # 数据集
    lyrics = LyricsDataset(corpus_index, 32)
    # 初始化模型
    model = TextGenerator()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化方法
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 训练轮数
    epoch = 200
    # 迭代打印
    iter_num = 300

    # 开始训练
    for epoch_idx in range(epoch):

        # 初始化数据加载器
        dataloader = DataLoader(lyrics, shuffle=True, batch_size=1)
        # 训练时间
        start = time.time()
        # 迭代次数
        iter_num = 0
        # 训练损失
        total_loss = 0.0

        for x, y in dataloader:

            # 初始化隐藏状态
            hidden = model.init_hidden()
            # 送入网络计算
            output, _ = model(x, hidden)
            # 计算损失
            # print(output.shape)
            # print(y.shape)
            loss = criterion(output.squeeze(), y.squeeze())
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            iter_num += 1
            total_loss += loss.item()


        info = 'epoch:%3s loss:%.5f time:%.2f' % \
               (epoch_idx,
                total_loss / iter_num,
                time.time() - start)

        print(info)

    # 模型保存
    torch.save(model.state_dict(), 'model/text-generator.pth')


# 预测函数
def predict(start_word, sentence_length):

    # 构建词典
    index_to_word, word_to_index, word_len, corpus_index = build_vocab()

    # 加载模型
    model = TextGenerator()
    model.load_state_dict(torch.load('model/text-generator.pth'))
    model.eval()

    # 初始化隐藏状态
    hidden = model.init_hidden()

    # 首先, 将 start_word 转换为索引
    word_idx = word_to_index[start_word]
    generate_sentence = [word_idx]
    for _ in range(sentence_length):
        output, hidden = model(torch.tensor([[word_idx]]), hidden)
        # 选择分数最大的词作为预测词
        word_idx = torch.argmax(output)
        generate_sentence.append(word_idx)

    # 最后, 将索引序列转换为词的序列
    for idx in generate_sentence:
        print(index_to_word[idx], end='')
    print()


if __name__ == '__main__':
    predict('分手', 50)
