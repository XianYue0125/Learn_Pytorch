import torch
import torch.nn as nn
import jieba   # pip install jieba


if __name__ == '__main__':

    text = '北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'

    # 1. 分词
    words = jieba.lcut(text)
    print(words)

    # 2. 构建词表
    index_to_word = {}  # 给定一个索引, 能够查询到该索引对应的词
    word_to_index = {}  # 给定一个词，能够查询到该词对应的索引
    # 去重
    unique_words = list(set(words))
    for idx, word in enumerate(unique_words):
        index_to_word[idx] = word
        word_to_index[word] = idx

    # 3. 构建词嵌入层
    embed = nn.Embedding(num_embeddings=len(index_to_word), embedding_dim=4)

    # 4. 将文本转换为词向量表示
    # print(index_to_word[0])
    # print(embed(torch.tensor(0)))
    # 将句子数值化
    for word in words:
        # 获得词对应的唯一的索引
        idx = word_to_index[word]
        # 根据唯一的索引找到该词的向量表示
        word_vec = embed(torch.tensor(idx))
        print('%3s\t' % word, word_vec)




