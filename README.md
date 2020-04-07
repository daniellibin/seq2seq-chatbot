
# seq2seq-chatbot
### **先上效果图**

<img src="./img/test.png" style="zoom:67%;" />

### 文件说明

#### **1.config.py参数配置文件**

主要进行模型超参数以及相关文件路径的配置

#### **2.DataProcessing.py 预处理文件**

主要进行语料库的处理工作，包括语料处理、编码索引、生成语料库的词向量文件emb等。

#### **3.read_vecor.py 修改词向量文件**

原始词向量是由维基百科语料word2vec训练得到的，现在要对原始词向量进行一定的修改，

主要加入了  

PAD = '</PAD>'  # 填充

UNK = '</UNK>'  # 未知

START = '</SOS>' # 开始

END = '</EOS>'  # 结束

这四个的词向量，随机生成（设置随机种子）。

- wiki.zh.text.vector 对应原始词向量
- word_vec.pkl 对应修改的词向量

#### **4.SequenceToSequence.py Seq2Seq模型**

#### **5.Train.py 训练文件**

运算只需要运行此文件即可

#### **6.RestfulAPI.py**

运行此文件，然后打开index.html，即可进行人机对话。

### **模型文件及相关数据文件请参考百度网盘：**

如果缺少相关数据或模型文件，请到这里下载。

| 文件名称                          | 解释                                                     |
| --------------------------------- | -------------------------------------------------------- |
| clean_chat_corpus/xiaohuangji.tsv | 小黄鸡训练语料                                           |
| model/                            | 训练好的模型文件，可直接加载                             |
| data/data.pkl                     | 原始语料预处理之后的数据                                 |
| data/wiki.zh.text.vector          | 原始词向量文件                                           |
| data/word_vec.pkl                 | 修改后的词向量文件                                       |
| data/emb.pkl                      | 根据语料库的词语抽取出的词向量文件，用于embedding_lookup |
| data/w2i.pkl                      | 词与索引对应的文件                                       |

链接：https://pan.baidu.com/s/1X2fixauTOE7RBkojBD90Pw 
提取码：yvxd 


### **详细介绍请参考博客：**

[1.基于seq2seq的中文聊天机器人（一）](https://blog.csdn.net/daniellibin/article/details/103290169)

[2.基于seq2seq的中文聊天机器人（二）](https://blog.csdn.net/daniellibin/article/details/103290395)

[3.基于seq2seq的中文聊天机器人（三）](https://blog.csdn.net/daniellibin/article/details/103290756)


