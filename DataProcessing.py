# encoding: utf-8

"""
    数据处理单元
    处理原始语料数据
    生成批训练数据
"""

import re
import os
import pickle
import collections
import jieba
import itertools
import random
import numpy as np

class DataUnit(object):

    # 特殊标签
    PAD = '<PAD>'  # 填充
    UNK = '<UNK>'  # 未知
    START = '<SOS>'
    END = '<EOS>'

    # 特殊标签的索引
    START_INDEX = 0
    END_INDEX =1
    UNK_INDEX = 2
    PAD_INDEX = 3

    def __init__(self, path, processed_path,
                 min_q_len, max_q_len,
                 min_a_len, max_a_len,
                 word2index_path, emb_path,
                 word_vec_path):
        """
            初始化函数，参数意义可查看CONFIG.py文件中的注释
        :param
        """
        self.path = path
        self.processed_path = processed_path
        self.word2index_path = word2index_path
        self.word_vec_path = word_vec_path
        self.emb_path = emb_path
        self.min_q_len = min_q_len
        self.max_q_len = max_q_len
        self.min_a_len = min_a_len
        self.max_a_len = max_a_len
        self.vocab_size = 0
        self.index2word = {}
        self.word2index = {}
        self.data = self.load_data()  # 加载处理好的语料
        self._fit_data_()       # 得到处理后语料库的所有词，并将其编码为索引值
        self.emb_write()

    def next_batch(self, batch_size):
        """
        生成一批训练数据
        :param batch_size: 每一批数据的样本数
        :return: 经过了填充处理的QA对
        """
        data_batch = random.sample(self.data, batch_size)
        batch = []
        for qa in data_batch:
            encoded_q = self.transform_sentence(qa[0])  # 将句子转化为索引，qa[0]代表question
            encoded_a = self.transform_sentence(qa[1])  # qa[1]代表answer


            # 填充句子
            q_len = len(encoded_q)
            encoded_q = encoded_q + [self.func_word2index(self.PAD)] * (self.max_q_len - q_len)

            encoded_a = encoded_a + [self.func_word2index(self.END)]
            a_len = len(encoded_a)
            encoded_a = encoded_a + [self.func_word2index(self.PAD)] * (self.max_a_len + 1 - a_len)

            batch.append((encoded_q, q_len, encoded_a, a_len))

        batch = zip(*batch)
        batch = [np.asarray(x) for x in batch]
        return batch   # 返回数组的格式[batch_size,4],4列分别为encoded_q, q_len, encoded_a, a_len

    def transform_sentence(self, sentence):
        """
        将句子转化为索引
        :param sentence:
        :return:
        """
        res = []

        for word in jieba.lcut(sentence):  # 未分词，直接按单字处理
            res.append(self.func_word2index(word))
        return res

    def transform_indexs(self, indexs):   # RestfulAPI中调用
        """
        将索引转化为句子,同时去除填充的标签
        :param indexs:索引序列
        :return:
        """
        res = []
        for index in indexs:
            if index == self.START_INDEX or index == self.PAD_INDEX \
                    or index == self.END_INDEX or index == self.UNK_INDEX:
                continue
            res.append(self.func_index2word(index))
        return ''.join(res)

    def _fit_data_(self):
        """
        得到处理后语料库的所有词，并将其编码为索引值
        :return:
        """
        if not os.path.exists(self.word2index_path):
            vocabularies = [x for x in self.data]  # x[0]为question,x[1]为answer
            self._fit_word_(itertools.chain(*vocabularies))   # itertools.chain可以把一组迭代对象串联起来，形成一个更大的迭代器;最终的格式为["q1","a1","q2","a12"......]
            with open(self.word2index_path, 'wb') as fw:
                pickle.dump(self.word2index, fw)
        else:
            with open(self.word2index_path, 'rb') as fr:
                self.word2index = pickle.load(fr)
            self.index2word = dict([(v,k) for k,v in self.word2index.items()])  # k为值，v 为索引  index2word例如('1',你)
        self.vocab_size = len(self.word2index)


    def load_data(self):
        """
        获取处理后的语料库
        :return:
        """
        if not os.path.exists(self.processed_path):
            data = self._extract_data()
            with open(self.processed_path, 'wb') as fw:
                pickle.dump(data, fw)
        else:
            with open(self.processed_path, 'rb') as fr:
                data = pickle.load(fr)
        # 根据CONFIG文件中配置的最大值和最小值问答对长度来进行数据过滤
        data = [x for x in data if self.min_q_len <= len(x[0]) <= self.max_q_len and self.min_a_len <= len(x[1]) <= self.max_a_len]
        return data

    def func_word2index(self, word):
        """
        将词转化为索引
        :param word:
        :return:
        """
        return self.word2index.get(word, self.word2index[self.UNK])  # 未知字符转换为self.UNK的索引

    def func_index2word(self, index):
        """
        将索引转化为词
        :param index:
        :return:
        """
        return self.index2word.get(index, self.UNK)  # 未知字符用UNK代替

    def jieba_sentence(self,vocabularies):
        groups = []
        for voc in vocabularies:
            groups.append(jieba.lcut(voc))

        return groups

    def _fit_word_(self, vocabularies, min_count=5, max_count=None):
        """
        将词表中所有的词转化为索引，过滤掉出现次数少于4次的词
        :param vocabularies:词表
        :return:
        """
        count = {}
        groups = self.jieba_sentence(vocabularies)
        for arr in groups:
            for a in arr:
                if a not in count:
                    count[a] = 0
                count[a] += 1

        if min_count is not None:
            count = {k: v for k, v in count.items() if v >= min_count}

        if max_count is not None:
            count = {k: v for k, v in count.items() if v <= max_count}

        index2word = [self.START] + [self.END] + [self.UNK] + [self.PAD] + [w for w in sorted(count.keys())]   # x[0]代表字符，x[1]代表出现的次数
        self.word2index = dict([(w, i) for i, w in enumerate(index2word)])  # i代表顺序，作为索引；w为字符   word2index例如 ('你'，1)
        self.index2word = dict([(i, w) for i, w in enumerate(index2word)])  # index2word ('1'，你)

    def emb_write(self):
        if not os.path.exists(self.emb_path):
            word_vec = pickle.load(open(self.word_vec_path, 'rb'))
            emb = np.zeros((len(self.word2index), len(word_vec['<SOS>'])))
            np.random.seed(1)
            for word, ind in self.word2index.items():
                if word in word_vec:
                    emb[ind] = word_vec[word]
                else:
                    emb[ind] = np.random.random(size=(300,)) - 0.5

            pickle.dump(emb,open(self.emb_path, 'wb'))


    def _regular_(self, sen):
        """
        句子规范化，主要是对原始语料的句子进行一些标点符号的统一
        :param sen:
        :return:
        """
        sen = sen.replace('/', '')  # 将语料库中的/替为空
        sen = re.sub(r'…{1,100}', '…', sen)
        sen = re.sub(r'\.{3,100}', '…', sen)
        sen = re.sub(r'···{2,100}', '…', sen)
        sen = re.sub(r',{1,100}', '，', sen)
        sen = re.sub(r'\.{1,100}', '。', sen)
        sen = re.sub(r'。{1,100}', '。', sen)
        sen = re.sub(r'\?{1,100}', '？', sen)
        sen = re.sub(r'？{1,100}', '？', sen)
        sen = re.sub(r'!{1,100}', '！', sen)
        sen = re.sub(r'！{1,100}', '！', sen)
        sen = re.sub(r'~{1,100}', '～', sen)
        sen = re.sub(r'～{1,100}', '～', sen)
        sen = re.sub(r'[“”]{1,100}', '"', sen)
        sen = re.sub('[^\w\u4e00-\u9fff"。，？！～·]+', '', sen)  # 把非汉字字符和全角字符替换为空
        sen = re.sub(r'[ˇˊˋˍεπのゞェーω]', '', sen)

        return sen

    def _good_line_(self, line):
        """
        判断一句话是否是好的语料,即判断汉字所占比例>=0.8，且数字字母<3个，其他字符<3个
        :param line:
        :return:
        """
        if len(line) == 0:
            return False
        ch_count = 0
        for c in line:
            # 中文字符范围
            if '\u4e00' <= c <= '\u9fff':
                ch_count += 1
        if ch_count / float(len(line)) >= 0.8 and len(re.findall(r'[a-zA-Z0-9]', ''.join(line))) < 3 \
                and len(re.findall(r'[ˇˊˋˍεπのゞェーω]', ''.join(line))) < 3:
            return True
        return False

    def _extract_data(self):
        res = []
        for maindir, subdir, file_name_list in os.walk(self.path):
            for filename in file_name_list:
                with open(self.path+filename, 'r', encoding='utf-8') as fr:
                    for line in fr:
                        if '\t' not in line:  # 错误处理的句子进行过滤
                            continue
                        q, a = line.split('\t')
                        q = self._regular_(q)
                        a = self._regular_(a)
                        if self._good_line_(q) and self._good_line_(a):
                            res.append((q, a))

        return res

    def __len__(self):  # 函数的重载
        """
        返回处理后的语料库中问答对的数量
        :return:
        """
        return len(self.data)


