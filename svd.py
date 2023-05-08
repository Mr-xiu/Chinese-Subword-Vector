import numpy as np


class SVD:
    """
    SVD类
    """

    def __init__(self, train_path: str = 'data/corpus.txt', vocab_max_size=10000):
        """
        初始化的方法
        :param train_path: 训练语料的位置
        :param vocab_max_size: 词表最大的大小
        """
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
        # 词表大小
        self.vocab_size = len(self.word2id_dict) if vocab_max_size >= len(self.word2id_dict) else vocab_max_size

        print(f'构建id完成，语料中共有{self.vocab_size}个词~')

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

    def build_svd_vector(self, save_path='model/svd.npy', vector_dim=100, window_size=5):
        """
        通过svd构建词向量的方法
        :param save_path: 词向量表保存的位置
        :param vector_dim: 词向量的维度
        :param window_size: 共现窗口的大小
        """
        # 初始化共现矩阵
        co_matrix = np.zeros((self.vocab_size, self.vocab_size), dtype='uint16')
        for line_corpus in self.corpus:
            # 从左到右，枚举每个中心点的位置
            for center_word_idx in range(len(line_corpus)):
                # 当前的中心词
                center_word = line_corpus[center_word_idx]
                if self.word2id_dict[center_word] >= self.vocab_size:
                    continue
                # 上下文词列表
                context_words_list = line_corpus[max(0, center_word_idx - window_size): center_word_idx] + line_corpus[
                                                                                                           center_word_idx + 1: center_word_idx + window_size + 1]
                # 更新共现矩阵
                for context_word in context_words_list:
                    if self.word2id_dict[context_word] < self.vocab_size:
                        co_matrix[self.word2id_dict[center_word], self.word2id_dict[context_word]] += 1
        print('构建共现矩阵完成，开始进行SVD分解~')
        # 对共现矩阵进行SVD分解，得到U、Σ和V矩阵
        U, S, V = np.linalg.svd(co_matrix)


        # 选择U矩阵的前50列作为词向量矩阵
        word_vectors = U[:, :vector_dim]
        np.save(save_path, np.array(word_vectors))


if __name__ == "__main__":
    svd = SVD(train_path='data/corpus.txt')
    svd.build_svd_vector(save_path='model/svd.npy', vector_dim=100, window_size=5)
