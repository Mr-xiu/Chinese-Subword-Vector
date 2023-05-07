import torch
from torch.utils.data import Dataset


class SkipGram(torch.nn.Module):
    """
    基于pytorch构建的SkipGram模型
    """

    def __init__(self, vocab_size, embedding_size, init_range=0.1):
        """
        SkipGram的初始化方法
        :param vocab_size: 传入词表的大小
        :param embedding_size: 嵌入向量的维度大小
        :param init_range: embedding层初始化权重的范围
        """
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size  # 词表大小
        self.embedding_size = embedding_size  # 嵌入向量的维度大小

        # 构造一个词向量参数
        # 参数的初始化方法为在[-init_scale, init_scale]区间进行均匀采样
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            padding_idx=None,
            max_norm=None,
            norm_type=2,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=torch.FloatTensor(vocab_size, embedding_size).uniform_(-init_range, init_range)
        )

        # 构造另外一个词向量参数
        self.embedding_out = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            padding_idx=None,
            max_norm=None,
            norm_type=2,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=torch.FloatTensor(vocab_size, embedding_size).uniform_(-init_range, init_range)
        )

    # 定义网络的前向计算逻辑
    def forward(self, center_words, target_words, label):
        """
        前向传播并计算loss
        :param center_words: tensor类型，表示中心词
        :param target_words: tensor类型，表示目标词
        :param label: tensor类型，表示词的样本标签（正样本为1，负样本为0）
        :return: 前向传播得到的loss
        """
        # 通过embedding参数，将mini-batch中的词转换为词向量
        center_words_emb = self.embedding(center_words)
        target_words_emb = self.embedding_out(target_words)

        # 通过点乘的方式计算中心词到目标词的输出概率
        word_sim = torch.sum(center_words_emb * target_words_emb, dim=-1)

        # 通过估计的输出概率定义损失函数
        # 包含sigmoid和cross entropy两步
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(word_sim, label.float())
        # 返回loss
        return loss


class SkipGramDataset(Dataset):
    def __init__(self, dataset):
        """
        初始化数据集的方法
        :param dataset: 传入的数据集
        """
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        读取数据集中的项目
        """
        center_word, target_word, label = self.dataset[index]
        return center_word, target_word, label
