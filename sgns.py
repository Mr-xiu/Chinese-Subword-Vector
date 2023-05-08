import math
import random

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from skip_gram import SkipGramDataset, SkipGram
import torch


class SGNS:
    """
    SGNS类，即基于 Skip-Gram with Negative Sampling (SGNS) 的汉语词向量处理类
    """

    def __init__(self, train_path: str = 'data/corpus.txt', window_size=2, negative_sample_num=4, embedding_size=100,
                 device='cpu', has_train=False):
        """
        初始化的方法
        :param train_path: 训练语料的位置
        :param window_size: window_size代表了window_size的大小，程序会根据window_size从左到右扫描整个语料
        :param negative_sample_num: negative_sample_num代表了对于每个正样本，我们需要随机采样多少负样本用于训练
        :param embedding_size: 嵌入向量的维度大小
        :param device: 当前使用的设备('cpu'或'cuda')
        :param has_train: 若为True，表示模型已经训练成功，不进行二次采样与构建dataloader（省内存）
        """
        self.embeddings = None
        self.dataset = None  # 数据集
        self.dataloader = None
        self.device = device
        self.corpus = []  # 语料列表（每一项为每一行的语料）
        # 先读取语料
        with open(train_path, 'r', encoding='UTF-8') as f:
            self.corpus = f.readlines()
            f.close()

        # 预处理语料
        self.preprocess_corpus()

        # 根据词频为每个词构建对应的ID
        self.id2word_dict = {}  # key为id，值为对应的word
        self.word2id_dict = {}  # key为word，值为对应的id
        self.freq_dict = {}  # key为word，值为对应的频率
        self.corpus_size = 0  # 语料中词的数量
        self.create_id()  # 开始构建
        self.vocab_size = len(self.word2id_dict)  # 词表大小
        print(f'构建id完成，语料中共有{self.vocab_size}个词~')

        if not has_train:
            # 进行二次采样
            self.subsampling()
            # 构建数据集
            self.build_data(window_size, negative_sample_num)

        self.model = SkipGram(self.vocab_size, embedding_size).to(self.device)

    def preprocess_corpus(self):
        """
        预处理语料的方法
        """
        # 符号表
        symbol_set = {'。', '，', '？', '！', '；', '：', '、', '（', '）', '「', '」', '“', '”', '‘', '’', '《', '》', '【', '】',
                      '…', '—', '～', '　', '.', ',', '?', '!', ';', ':', '(', ')', '"', '"', '\'', '\'', '<', '>', '[',
                      ']', '...', '~', '*'}

        for i in range(len(self.corpus)):
            line_list = self.corpus[i].strip('\n').split()
            # print('，'.join(line_list))
            self.corpus[i] = []
            for word in line_list:
                # 除去符号表中的元素
                if word not in symbol_set:
                    self.corpus[i].append(word)
            # print('，'.join(corpus[i]))

        return self.corpus

    def create_id(self):
        """
        为语料中的每个词根据出现的频率构建ID
        """
        # 遍历每行的语料
        for line_corpus in self.corpus:
            # 遍历每个词
            for word in line_corpus:
                self.corpus_size += 1
                if word not in self.freq_dict:
                    self.freq_dict[word] = 1
                else:
                    self.freq_dict[word] += 1
        # 按照词频排序
        sort_dict = sorted(self.freq_dict.items(), key=lambda x: x[1], reverse=True)

        # 构建ID（频率越高id越小）
        for word, _ in sort_dict:
            my_id = len(self.word2id_dict)
            # print(word, _)
            self.word2id_dict[word] = my_id
            self.id2word_dict[my_id] = word

    def subsampling(self):
        """
        二次采样算法，二次采样的同时将语料转换为id序列
        """
        for i in range(len(self.corpus)):
            line_corpus = self.corpus[i]
            self.corpus[i] = []
            for word in line_corpus:

                drop = random.uniform(0, 1) < 1 - math.sqrt(1e-4 / self.freq_dict[word] * self.corpus_size)
                # 若不丢弃
                if not drop:
                    self.corpus[i].append(self.word2id_dict[word])

    def build_data(self, window_size, negative_sample_num):
        """
        采样数据集
        :param window_size: window_size代表了window_size的大小，程序会根据window_size从左到右扫描整个语料
        :param negative_sample_num: negative_sample_num代表了对于每个正样本，我们需要随机采样多少负样本用于训练
        :return:
        """
        dataset = []
        for line_corpus in self.corpus:
            # 从左到右，枚举每个中心点的位置
            for center_word_idx in range(len(line_corpus)):
                # 当前的中心词
                center_word = line_corpus[center_word_idx]
                # 以中心词为中心，将左右两侧在窗口内的词均看为正样本
                positive_word_range = (
                    max(0, center_word_idx - window_size), min(len(line_corpus) - 1, center_word_idx + window_size))
                positive_word_set = set(
                    [line_corpus[idx] for idx in range(positive_word_range[0], positive_word_range[1] + 1) if
                     idx != center_word_idx])
                # 对于每个正样本，随机采样negative_sample_num个负样本进行训练
                for positive_word in positive_word_set:
                    # 先加入正样本
                    dataset.append((center_word, positive_word, 1))

                    # 开始负采样
                    i = 0
                    while i < negative_sample_num:
                        negative_word = random.randint(0, self.vocab_size - 1)
                        if negative_word not in positive_word_set:
                            # 加入负样本
                            dataset.append((center_word, negative_word, 0))
                            i += 1

        self.dataset = SkipGramDataset(dataset)  # 设置数据集
        self.dataloader = DataLoader(self.dataset, batch_size=512, shuffle=True)

    def train(self, epoch_num, save_path='model/SGNS.pth'):
        """
        开始训练的方法，调用该方法进行训练
        :param epoch_num: 训练的轮数
        :param save_path: 模型保存的位置
        """
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001)
        loss = -1
        for epoch in range(epoch_num):
            for _, (center_word, target_word, is_positive) in enumerate(tqdm(self.dataloader)):
                center_word = center_word.to(self.device)
                target_word = target_word.to(self.device)
                is_positive = is_positive.to(self.device)
                loss = self.model(center_word, target_word, is_positive)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'第{epoch + 1}轮训练结束,loss为{loss:>7f}')
        torch.save(self.model.state_dict(), save_path)
        self.embeddings = self.model.embedding.to('cpu')
        self.embeddings = self.embeddings.weight.data.numpy()

    def load_model(self, model_path='model/SGNS.pth'):
        """
        从磁盘中读取模型
        :param model_path: 模型的位置
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.embeddings = self.model.embedding.to('cpu')
        self.embeddings = self.embeddings.weight.data.numpy()

    def get_cos_sim(self, word1, word2):
        """
        计算传入词余弦相似度的方法
        :param word1: 第一个词
        :param word2: 第二个词
        :return: 这两个词的余弦相似度
        """
        # 如果词不在词典中，就忽略
        if (word1 not in self.word2id_dict) or (word2 not in self.word2id_dict):
            return 0
        word1_vec = self.embeddings[self.word2id_dict[word1]]
        word2_vec = self.embeddings[self.word2id_dict[word2]]
        cos_sim = np.dot(word1_vec, word2_vec) / (np.linalg.norm(word1_vec) * np.linalg.norm(word2_vec))
        return cos_sim


def test_sgns(has_train=True, epoch_num=5, model_path='model/SGNS.pth', test_path='data/pku_sim_test.txt',
              result_path='data/sgns_result.txt'):
    """
    在pku_sim_test.txt中测试训练模型表现的方法
    :param has_train: 若为True，表示模型已经训练成功，不进行二次采样与构建dataloader（省内存）
    :param epoch_num: 训练的轮数
    :param model_path: 模型的位置
    :param test_path: 测试文件的位置
    :param result_path: 测试结果文件保存的位置
    """
    if not has_train:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'device:{device}')
        my_sgns = SGNS(device='cpu', has_train=False)
        my_sgns.train(epoch_num, save_path=model_path)
    else:
        # 读取模型
        my_sgns = SGNS(device='cpu', has_train=True)
        my_sgns.load_model(model_path)

    # 读取测试文本
    with open(test_path, 'r', encoding='UTF-8') as f:
        test_lines = f.readlines()
        f.close()
    f = open(result_path, 'w', encoding='UTF-8')
    # 开始按行测试
    for i in range(len(test_lines)):
        line = test_lines[i].strip('\n').split('\t')
        if len(line) == 0:
            continue
        word1 = line[0]
        word2 = line[1]
        sim_sgns = my_sgns.get_cos_sim(word1, word2)
        f.write(f'{word1}\t{word2}\t{sim_sgns:.4f}\n')
    f.close()


if __name__ == "__main__":
    test_sgns(test_path='data/pku_sim_test.txt')
