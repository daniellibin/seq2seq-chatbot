# encoding: utf-8

"""
    SequenceToSequence模型
    定义了模型编码器、解码器、优化器、训练、预测
"""

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell, LSTMStateTuple, DropoutWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper, TrainingHelper, \
    BasicDecoder, BeamSearchDecoder
from tensorflow import layers
from DataProcessing import DataUnit
from tensorflow.python.ops import array_ops


class Seq2Seq(object):

    def __init__(self, hidden_size, cell_type,
                 layer_size, batch_size,
                 encoder_vocab_size, decoder_vocab_size,
                 embedding_dim, share_embedding,
                 max_decode_step, max_gradient_norm,
                 learning_rate, decay_step,
                 min_learning_rate, bidirection,
                 beam_width,
                 mode
                 ):
        """
        初始化函数
        """
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.layer_size = layer_size
        self.batch_size = batch_size
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.embedding_dim = embedding_dim
        self.share_embedding = share_embedding
        self.max_decode_step = max_decode_step
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate = learning_rate
        self.decay_step = decay_step
        self.min_learning_rate = min_learning_rate
        self.bidirection = bidirection
        self.beam_width = beam_width
        self.mode = mode
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.build_model()

    def build_model(self):
        """
        构建完整的模型
        :return:
        """
        self.init_placeholder()
        self.embedding()
        encoder_outputs, encoder_state = self.build_encoder()
        self.build_decoder(encoder_outputs, encoder_state)
        if self.mode == 'train':
            self.build_optimizer()
        self.saver = tf.train.Saver()

    def init_placeholder(self):
        """
        定义各个place_holder
        :return:
        """
        self.encoder_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, shape=[self.batch_size, ], name='encoder_inputs_length')
        self.keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')
        if self.mode == 'train':
            self.decoder_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='decoder_inputs')
            self.decoder_inputs_length = tf.placeholder(tf.int32, shape=[self.batch_size, ], name='decoder_inputs_length')
            self.decoder_start_token = tf.ones(shape=(self.batch_size, 1), dtype=tf.int32) * DataUnit.START_INDEX
            self.decoder_inputs_train = tf.concat([self.decoder_start_token, self.decoder_inputs], axis=1)


    def feed_embedding(self, sess, encoder=None, decoder=None):
        """加载预训练好的embedding
        """

        if encoder is not None:
            sess.run(self.encoder_embeddings_init,
                     {self.encoder_embeddings_placeholder: encoder})

        if decoder is not None:
            sess.run(self.decoder_embeddings_init,
                     {self.decoder_embeddings_placeholder: decoder})

    def embedding(self):
        """
        词嵌入操作
        :param share:编码器和解码器是否共用embedding
        :return:
        """
        with tf.variable_scope('embedding'):

            self.encoder_embeddings_placeholder = tf.placeholder(
                tf.float32,
                (self.encoder_vocab_size, self.embedding_dim)
            )

            self.decoder_embeddings_placeholder = tf.placeholder(
                tf.float32,
                (self.decoder_vocab_size, self.embedding_dim)
            )

            self.encoder_embeddings = tf.Variable(
                tf.constant(0.0,shape=(self.encoder_vocab_size, self.embedding_dim)),
                trainable=True,
                name='embeddings'
            )

            self.decoder_embeddings = tf.Variable(
                tf.constant(0.0, shape=(self.decoder_vocab_size,self.embedding_dim)),
                trainable=True,
                name='embeddings'
            )

            self.encoder_embeddings_init = self.encoder_embeddings.assign(self.encoder_embeddings_placeholder)
            self.decoder_embeddings_init = self.decoder_embeddings.assign(self.decoder_embeddings_placeholder)

            '''

            if not self.share_embedding:
                self.decoder_embeddings_init = self.decoder_embeddings.assign(self.decoder_embeddings_placeholder)
            else:
                self.decoder_embeddings_init = self.decoder_embeddings.assign(self.encoder_embeddings_placeholder)
            '''

    def one_cell(self, hidden_size, cell_type):
        """
        一个神经元
        :return:
        """
        if cell_type == 'gru':
            c = GRUCell
        else:
            c = LSTMCell
        cell = c(hidden_size)  # hidden_size rnn神经元单元的状态数
        cell = DropoutWrapper(   # dropout防止过拟合
            cell,
            dtype=tf.float32,
            output_keep_prob=self.keep_prob,
        )
        cell = ResidualWrapper(cell) # ResidualWrapper是RNNCell的实例，它对RNNCell进行了封装
        return cell

    def build_encoder_cell(self, hidden_size, cell_type, layer_size):
        """
        构建编码器所有层
        :param hidden_size:
        :param cell_type:
        :param layer_size:
        :return:
        """
        cells = [self.one_cell(hidden_size, cell_type) for _ in range(layer_size)]
        return MultiRNNCell(cells)

    def build_encoder(self):
        """
        构建完整编码器
        :return:
        """
        with tf.variable_scope('encoder'):
            encoder_cell = self.build_encoder_cell(self.hidden_size, self.cell_type, self.layer_size)
            # self.encoder_embeddings可以理解为词向量词典，存储vocab_size个大小为embedding_size的词向量，随机初始化为正态分布的值；
            # tf.nn.embedding_lookup（params, ids）函数的用法主要是选取一个张量里面索引对应的元素;params可以是张量也可以是数组等，id就是对应的索引。
            # [self.encoder_inputs,self.encoder_vocab_size]*[self.encoder_vocab_size, self.embedding_dim]=[self.encoder_inputs,self.embedding_dim]
            encoder_inputs_embedded = tf.nn.embedding_lookup(self.encoder_embeddings, self.encoder_inputs)
            encoder_inputs_embedded = layers.dense(encoder_inputs_embedded,   # 全连接层  相当于添加一个层
                                                   self.hidden_size,
                                                   use_bias=False,
                                                   name='encoder_residual_projection')
            initial_state = encoder_cell.zero_state(self.batch_size, dtype=tf.float32) # 初始化值
            if self.bidirection:   # 双向RNN
                encoder_cell_bw = self.build_encoder_cell(self.hidden_size, self.cell_type, self.layer_size)
                '''
                outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和后向cell输出tensor组成的二元组。
                output_states为(output_state_fw, output_state_bw)，包含了前向和后向最后的隐藏状态的组成的二元组。 
                output_state_fw和output_state_bw的类型为LSTMStateTuple。 
                LSTMStateTuple由（c，h）组成，分别代表memory cell和hidden state。
                '''
                ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_state, encoder_bw_state)) \
                    = tf.nn.bidirectional_dynamic_rnn(
                    cell_bw=encoder_cell_bw,  # 后向RNN
                    cell_fw=encoder_cell,     # 前向RNN
                    inputs=encoder_inputs_embedded,  # 输入
                    sequence_length=self.encoder_inputs_length, # 输入序列的实际长度
                    dtype=tf.float32,
                    swap_memory=True)

                encoder_outputs = tf.concat(
                    (encoder_bw_outputs, encoder_fw_outputs), 2)  # 在第二个维度拼接
                encoder_final_state = []
                # output_state_fw和output_state_bw的类型为LSTMStateTuple。
                # LSTMStateTuple由（c，h）组成，分别代表memory cell和hidden state。
                for i in range(self.layer_size):
                    c_fw, h_fw = encoder_fw_state[i]
                    c_bw, h_bw = encoder_bw_state[i]
                    c = tf.concat((c_fw, c_bw), axis=-1)  # 在最高的维度进行拼接
                    h = tf.concat((h_fw, h_bw), axis=-1)
                    encoder_final_state.append(LSTMStateTuple(c=c, h=h))
                encoder_final_state = tuple(encoder_final_state)
            else:
                encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                    cell=encoder_cell,
                    inputs=encoder_inputs_embedded,
                    sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32,
                    initial_state=initial_state,
                    swap_memory=True)

            return encoder_outputs, encoder_final_state

    def build_decoder_cell(self, encoder_outputs, encoder_final_state,
                           hidden_size, cell_type, layer_size):
        """
        构建解码器所有层
        :param encoder_outputs:
        :param encoder_state:
        :param hidden_size:
        :param cell_type:
        :param layer_size:
        :return:
        """
        sequence_length = self.encoder_inputs_length
        if self.mode == 'decode':
            encoder_outputs = tf.contrib.seq2seq.tile_batch(   # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份
                encoder_outputs, multiplier=self.beam_width)
            encoder_final_state = tf.contrib.seq2seq.tile_batch(
                encoder_final_state, multiplier=self.beam_width)
            sequence_length = tf.contrib.seq2seq.tile_batch(
                sequence_length, multiplier=self.beam_width)

        if self.bidirection:  # 编码器是否使用双向rnn
            cell = MultiRNNCell([self.one_cell(hidden_size * 2, cell_type) for _ in range(layer_size)])
        else:
            cell = MultiRNNCell([self.one_cell(hidden_size, cell_type) for _ in range(layer_size)])

        # 使用attention机制
        self.attention_mechanism = BahdanauAttention(
            num_units=self.hidden_size,
            memory=encoder_outputs,
            memory_sequence_length=sequence_length
        )

        def cell_input_fn(inputs, attention):
            mul = 2 if self.bidirection else 1
            attn_projection = layers.Dense(self.hidden_size * mul,
                                           dtype=tf.float32,
                                           use_bias=False,
                                           name='attention_cell_input_fn')
            return attn_projection(array_ops.concat([inputs, attention], -1))

        cell = AttentionWrapper(
            cell=cell,  # rnn cell实例，可以是单个cell，也可以是多个cell stack后的mutli layer rnn
            attention_mechanism=self.attention_mechanism,  # attention mechanism的实例，此处为BahdanauAttention
            attention_layer_size=self.hidden_size,  # 用来控制我们最后生成的attention是怎么得来;如果不是None，则在调用_compute_attention方法时，得到的加权和向量还会与output进行concat，然后再经过一个线性映射，变成维度为attention_layer_size的向量
            cell_input_fn=cell_input_fn,  # input送入decoder cell的方式，默认是会将input和上一步计算得到的attention拼接起来送入decoder cell,
            name='Attention_Wrapper'
        )
        if self.mode == 'decode':
            decoder_initial_state = cell.zero_state(batch_size=self.batch_size * self.beam_width,
                                                    dtype=tf.float32).clone(
                cell_state=encoder_final_state)
        else:
            decoder_initial_state = cell.zero_state(batch_size=self.batch_size,
                                                    dtype=tf.float32).clone(
                cell_state=encoder_final_state)
        return cell, decoder_initial_state

    def build_decoder(self, encoder_outputs, encoder_final_state):
        """
        构建完整解码器
        :return:
        """
        with tf.variable_scope("decode"):
            decoder_cell, decoder_initial_state = self.build_decoder_cell(encoder_outputs, encoder_final_state,
                                                                          self.hidden_size, self.cell_type,
                                                                          self.layer_size)
            # 输出层投影
            decoder_output_projection = layers.Dense(self.decoder_vocab_size, dtype=tf.float32,
                                                     use_bias=False,
                                                     kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                                        stddev=0.1),
                                                     name='decoder_output_projection')
            if self.mode == 'train':
                # 训练模式
                decoder_inputs_embdedded = tf.nn.embedding_lookup(self.decoder_embeddings, self.decoder_inputs_train)
                '''
                TrainingHelper用于train阶段，next_inputs方法一样也接收outputs与sample_ids，但是只是从初始化时的inputs返回下一时刻的输入。
                TrainingHelper
                __init__( inputs, sequence_length, time_major=False, name=None )
                - inputs: A (structure of) input tensors.
                - sequence_length: An int32 vector tensor.
                - time_major: Python bool. Whether the tensors in inputs are time major. If False (default), they are assumed to be batch major.
                - name: Name scope for any created operations.
                inputs：对应Decoder框架图中的embedded_input，time_major=False的时候，inputs的shape就是[batch_size, sequence_length, embedding_size] ，time_major=True时，inputs的shape为[sequence_length, batch_size, embedding_size]
                sequence_length：这个文档写的太简略了，不过在源码中可以看出指的是当前batch中每个序列的长度(self._batch_size = array_ops.size(sequence_length))。
                time_major：决定inputs Tensor前两个dim表示的含义
                name：如文档所述
                '''
                training_helper = TrainingHelper(
                    inputs=decoder_inputs_embdedded,
                    sequence_length=self.decoder_inputs_length,
                    name='training_helper'
                )
                '''
                BasicDecoder的作用就是定义一个封装了decoder应该有的功能的实例，根据Helper实例的不同，这个decoder可以实现不同的功能，比如在train的阶段，不把输出重新作为输入，而在inference阶段，将输出接到输入。
                BasicDecoder
                __init__( cell, helper, initial_state, output_layer=None )
                - cell: An RNNCell instance.
                - helper: A Helper instance.
                - initial_state: A (possibly nested tuple of…) tensors and TensorArrays. The initial state of the RNNCell.
                - output_layer: (Optional) An instance of tf.layers.Layer, i.e., tf.layers.Dense. Optional layer to apply to the RNN output prior to storing the result or sampling.
                cell：在这里就是一个多层LSTM的实例，与定义encoder时无异
                helper：这里只是简单说明是一个Helper实例，第一次看文档的时候肯定还不知道这个Helper是什么，不用着急，看到具体的Helper实例就明白了
                initial_state：encoder的final state，类型要一致，也就是说如果encoder的final state是tuple类型(如LSTM的包含了cell state与hidden state)，那么这里的输入也必须是tuple。直接将encoder的final_state作为这个参数输入即可
                output_layer：对应的就是框架图中的Dense_Layer，只不过文档里写tf.layers.Dense，但是tf.layers下只有dense方法，Dense的实例还需要from tensorflow.python.layers.core import Dense。
                '''
                training_decoder = BasicDecoder(decoder_cell, training_helper,
                                                decoder_initial_state, decoder_output_projection)
                max_decoder_length = tf.reduce_max(self.decoder_inputs_length)
                '''
                首先tf.contrib.seq2seq.dynamic_decode主要作用是接收一个Decoder类，然后依据Encoder进行解码，实现序列的生成（映射）。
                其中，这个函数主要的一个思想是一步一步地调用Decoder的step函数（该函数接收当前的输入和隐层状态会生成下一个词），实现最后的一句话的生成。该函数类似tf.nn.dynamic_rnn。
                '''
                training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                                  maximum_iterations=max_decoder_length)
                '''
                tf.sequence_mask函数返回的一个mask张量。经过tf.Session()打印可以得到一个array数据。
                decoder_inputs_length范围内的数据用1填充，[decoder_inputs_length,max_decoder_length]区间用0填充
                '''
                self.masks = tf.sequence_mask(self.decoder_inputs_length, maxlen=max_decoder_length, dtype=tf.float32,
                                              name='masks')
                '''
                tf.contrib.seq2seq.sequence_loss可以直接计算序列的损失函数，重要参数：
                logits：尺寸[batch_size, sequence_length, num_decoder_symbols]
                targets：尺寸[batch_size, sequence_length]，不用做one_hot。
                weights：[batch_size, sequence_length]，即mask，滤去padding的loss计算，使loss计算更准确。
                '''
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=training_decoder_output.rnn_output,
                                                             targets=self.decoder_inputs,
                                                             weights=self.masks,  # mask，滤去padding的loss计算，使loss计算更准确。
                                                             average_across_timesteps=True,
                                                             average_across_batch=True
                                                             )
            else:
                # 预测模式
                start_token = [DataUnit.START_INDEX] * self.batch_size
                end_token = DataUnit.END_INDEX
                '''
                BeamSearchDecoder             
                cell: An RNNCell instance.
                embedding: A callable that takes a vector tensor of ids (argmax ids), or the params argument for embedding_lookup.
                start_tokens: int32 vector shaped [batch_size], the start tokens.
                end_token: int32 scalar, the token that marks end of decoding.
                initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
                beam_width: Python integer, the number of beams.
                output_layer: (Optional) An instance of tf.keras.layers.Layer, i.e., tf.keras.layers.Dense. Optional layer to apply to the RNN output prior to storing the result or sampling.
                length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
                coverage_penalty_weight: Float weight to penalize the coverage of source sentence. Disabled with 0.0.
                reorder_tensor_arrays: If True, TensorArrays' elements within the cell state will be reordered according to the beam search path. 
                If the TensorArray can be reordered, the stacked form will be returned. Otherwise, 
                the TensorArray will be returned as is. Set this flag to False if the cell state contains TensorArrays that are not amenable to reordering.   
                '''
                inference_decoder = BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=lambda x: tf.nn.embedding_lookup(self.decoder_embeddings, x),
                    start_tokens=start_token,
                    end_token=end_token,
                    initial_state=decoder_initial_state,
                    beam_width=self.beam_width,
                    output_layer=decoder_output_projection
                )
                '''
                首先tf.contrib.seq2seq.dynamic_decode主要作用是接收一个Decoder类，然后依据Encoder进行解码，实现序列的生成（映射）。
                其中，这个函数主要的一个思想是一步一步地调用Decoder的step函数（该函数接收当前的输入和隐层状态会生成下一个词），实现最后的一句话的生成。该函数类似tf.nn.dynamic_rnn。
                 '''
                inference_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                                   maximum_iterations=self.max_decode_step)
                self.decoder_pred_decode = inference_decoder_output.predicted_ids
                self.decoder_pred_decode = tf.transpose(
                    self.decoder_pred_decode,
                    perm=[0, 2, 1]
                )

    def check_feeds(self, encoder_inputs, encoder_inputs_length,
                    decoder_inputs, decoder_inputs_length, keep_prob, decode):
        """
            检查输入,返回输入字典
        """
        input_batch_size = encoder_inputs.shape[0]
        assert input_batch_size == encoder_inputs_length.shape[0], 'encoder_inputs 和 encoder_inputs_length的第一个维度必须一致'
        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            assert target_batch_size == input_batch_size, 'encoder_inputs 和 decoder_inputs的第一个维度必须一致'
            assert target_batch_size == decoder_inputs_length.shape[0], 'decoder_inputs 和 decoder_inputs_length的第一个维度必须一致'

        input_feed = {self.encoder_inputs.name: encoder_inputs, self.encoder_inputs_length.name: encoder_inputs_length}
        input_feed[self.keep_prob.name] = keep_prob

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length
        return input_feed

    def build_optimizer(self):
        """
        构建优化器
        :return:
        """
        learning_rate = tf.train.polynomial_decay(self.learning_rate, self.global_step,
                                                  self.decay_step, self.min_learning_rate, power=0.5)
        self.current_learning_rate = learning_rate
        trainable_params = tf.trainable_variables()
        '''
        tf.gradients(
        ys,
        xs,
        grad_ys=None,
        name='gradients',
        stop_gradients=None,)
        1. xs和ys可以是一个张量，也可以是张量列表，tf.gradients(ys,xs) 实现的功能是求ys（如果ys是列表，那就是ys中所有元素之和）关于xs的导数（如果xs是列表，那就是xs中每一个元素分别求导），
        返回值是一个与xs长度相同的列表。例如ys=[y1,y2,y3], xs=[x1,x2,x3,x4]，那么tf.gradients(ys,xs)=[d(y1+y2+y3)/dx1,d(y1+y2+y3)/dx2,d(y1+y2+y3)/dx3,d(y1+y2+y3)/dx4].具体例子见下面代码第16-17行。
        2. grad_ys 是ys的加权向量列表，和ys长度相同，当grad_ys=[q1,q2,g3]时，tf.gradients(ys,xs，grad_ys)=[d(g1*y1+g2*y2+g3*y3)/dx1,d(g1*y1+g2*y2+g3*y3)/dx2,d(g1*y1+g2*y2+g3*y3)/dx3,d(g1*y1+g2*y2+g3*y3)/dx4].具体例子见下面代码第19-21行。
        3. stop_gradients使得指定变量不被求导，即视为常量，具体的例子见官方例子，此处省略。
        '''
        gradients = tf.gradients(self.loss, trainable_params)
        # 优化器
        self.opt = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        )
        # 梯度裁剪
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm
        )
        # 更新梯度
        self.update = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params),
            global_step=self.global_step
        )

    def train(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length, keep_prob):
        """
        训练模型
        :param sess:
        :return:
        """
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length, keep_prob,
                                      False)
        output_feed = [
            self.update, self.loss,
            self.current_learning_rate
        ]
        _, cost, lr = sess.run(output_feed, input_feed)
        return cost, lr

    def predict(self, sess, encoder_inputs, encoder_inputs_length):
        """
        预测
        :return:
        """
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      None, None, 1, True)
        pred = sess.run(self.decoder_pred_decode, input_feed)
        return pred[0]

    def save(self, sess, save_path='model/chatbot_model.ckpt'):
        """
        保存模型
        :return:
        """
        self.saver.save(sess, save_path=save_path)

    def load(self, sess, save_path='model/chatbot_model.ckpt'):
        """
        加载模型
        """
        self.saver.restore(sess, save_path)
