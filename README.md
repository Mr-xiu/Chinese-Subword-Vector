# Chinese-Subword-Vector
bupt nlp第二次作业：分别基于SVD分解以及基于SGNS两种方法构建汉语子词向量并进行评测
具体说明：



1、词表：为第一次编程作业中获得的子词词表中的子词构建向量。



2、语料：corpus.txt，语料采用第一次编程作业中的训练和测试集的并集，如果实在计算资源不够，可选择一个子集，但是评测时不区分。



3、基于SVD分解的方法：获取高维distributional表示时K=5，SVD降维后的维数自定，获得子词向量vec_sta。之后基于该向量计算pku_sim_test.txt中同一行中两个子词的余弦相似度sim_svd。当pku_sim_test.txt中某一个词没有获得向量时(该词未出现在该语料中)，令其所在行的两个词之间的sim_svd=0。



4、基于SGNS的方法：SGNS方法中窗口K=2，子词向量维数自定，获得向量vec_sgns。之后基于该子词向量计算pku_sim_test.txt中同一行中两个词的余弦相似度sim_sgns。当pku_sim_test.txt中某一个词没有获得向量时(该词未出现在该语料中)，令其所在行的两个词之间的sim_sgns=0。



5、两种方法的结果输出要求(因为是机器判定，请一定按如下格式输出)：



5.1 保持pku_sim_test.txt编码(utf-8)不变，保持原文行序不变



5.2 每行在行末加一个tab符之后写入该行两个词的sim_sv，再加一个tab符之后写入该行两个词的sim_sgns。



5.3 输出文件命名方式：学号。



6、所有输出文本均采用Unicode(UTF-8)编码



7、算法采用Python (3.0以上版本) 实现



8、计算资源：Google Colab 的GPU 可以考虑：免费的以及  9.9美元包月的
