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
        self.word_vectors = None
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
        self.word_vectors = U[:, :vector_dim]

        np.save(save_path, np.array(self.word_vectors))

    def load_svd_vector(self, model_path='model/svd.npy'):
        self.word_vectors = np.load(model_path)
        print('模型加载成功！')

    def get_cos_sim(self, word1, word2):
        """
        计算传入词余弦相似度的方法
        :param word1: 第一个词
        :param word2: 第二个词
        :return: 这两个词的余弦相似度
        """
        # 如果词不在词典中，就忽略
        if (word1 not in self.word2id_dict) or (word2 not in self.word2id_dict) or (self.word2id_dict[word1] >= self.vocab_size) or (
                self.word2id_dict[word2] >= self.vocab_size):
            return 0
        word1_vec = self.word_vectors[self.word2id_dict[word1]]
        word2_vec = self.word_vectors[self.word2id_dict[word2]]
        cos_sim = np.dot(word1_vec, word2_vec) / (np.linalg.norm(word1_vec) * np.linalg.norm(word2_vec))
        return cos_sim


def get_svd_result(has_train=True, vocab_max_size=10000, vector_dim=100, window_size=5, model_path='model/svd.npy',
                   test_path='data/pku_sim_test.txt',
                   result_path='data/svd_result.txt'):
    """
    在pku_sim_test.txt中测试训练模型表现的方法
    :param has_train: 若为True，表示模型已经训练成功，只需在内存中加载svd词向量矩阵
    :param vocab_max_size: 词表最大的大小
    :param vector_dim: 词向量的维度
    :param window_size: 共现窗口的大小
    :param model_path: svd词向量矩阵的位置
    :param test_path: 测试文件的位置
    :param result_path: 测试结果文件保存的位置
    """
    if not has_train:
        svd = SVD(train_path='data/corpus.txt', vocab_max_size=vocab_max_size)
        svd.build_svd_vector(save_path=model_path, vector_dim=vector_dim, window_size=window_size)
    else:
        # 读取模型
        svd = SVD(train_path='data/corpus.txt', vocab_max_size=vocab_max_size)
        svd.load_svd_vector(model_path)

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
        sim_sgns = svd.get_cos_sim(word1, word2)
        f.write(f'{word1}\t{word2}\t{sim_sgns:.4f}\n')
    f.close()


if __name__ == "__main__":
    get_svd_result(has_train=True, test_path='data/pku_sim_test.txt', vocab_max_size=20000)
