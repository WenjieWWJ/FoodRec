"""
这个文件中第一次attention_weights的循环和后面的不同，第一次循环将Label Matrix的维度由(num_labels, embed_size)改变为
(None, num_labels, embed_size),最终计算logits时使用的Label Matrix维度也是(None, num_labels, embed_size)
"""

import tensorflow as tf
import numpy as np
import copy


class Model:
    def __init__(self, FLAGS, filter_sizes, vocab_size, clip_gradients=5.0):
        self.num_classes = FLAGS.num_labels
        self.batch_size = FLAGS.batch_size
        self.sequence_length = FLAGS.sentence_len
        self.vocab_size = vocab_size
        self.embed_size = FLAGS.embed_size
        self.learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False,
                                         name="learning_rate")  # ADD learning_rate
        self.filter_sizes = filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters = FLAGS.num_filters
        self.num_filters_total = self.num_filters * len(filter_sizes)  # how many filters totally.
        self.multi_label_flag = FLAGS.multi_label_flag
        self.clip_gradients = clip_gradients
        self.num_labels = FLAGS.num_labels
        self.decay_steps, self.decay_rate = FLAGS.decay_steps, FLAGS.decay_rate
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.attention_loops = FLAGS.attention_loops

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y_multilabel")
        self.is_training_flag = tf.placeholder(tf.bool, name="is_training_flag")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.instantiate_weights()
        self.logits = self.inference()
        self.loss_val = self.loss_multilabel()
        self.train_op = self.train()

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"):
            self.User_Embedding = tf.get_variable("user_embedding", shape=[self.vocab_size, self.embed_size],
                                                  initializer=self.initializer)
            self.Label_Matrix = tf.get_variable("label_embedding", shape=[self.num_labels, self.embed_size],
                                                initializer=self.initializer)

            self.left_side_first_word = tf.get_variable("left_side_first_word",
                                                        shape=[self.batch_size, self.embed_size],
                                                        initializer=self.initializer)
            self.right_side_last_word = tf.get_variable("right_side_last_word",
                                                        shape=[self.batch_size, self.embed_size],
                                                        initializer=self.initializer)
            self.W_l = tf.get_variable("W_l", shape=[self.embed_size, self.embed_size], initializer=self.initializer)
            self.W_r = tf.get_variable("W_r", shape=[self.embed_size, self.embed_size], initializer=self.initializer)
            self.W_sl = tf.get_variable("W_sl", shape=[self.embed_size, self.embed_size], initializer=self.initializer)
            self.W_sr = tf.get_variable("W_sr", shape=[self.embed_size, self.embed_size], initializer=self.initializer)
            self.b = tf.get_variable("b", [self.embed_size])
            self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total, self.embed_size],
                                                initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection",
                                                shape=[self.embed_size])  # [label_size] #ADD 2017.06.09

    def get_context_left(self, context_left, embedding_previous):
        """
        :param context_left:
        :param embedding_previous:
        :return: output:[None,embed_size]
        """
        left_c = tf.matmul(context_left, self.W_l)
        # shape:[batch_size,embed_size]<---------context_left:[batch_size,embed_size];W_l:[embed_size,embed_size]
        left_e = tf.matmul(embedding_previous, self.W_sl)
        # shape:[batch_size,embed_size]<---------embedding_previous;[batch_size,embed_size];W_sl:[embed_size, embed_size]
        left_h = left_c + left_e
        # shape:[batch_size,embed_size]
        # context_left=self.activation(left_h) #shape:[batch_size,embed_size] #TODO
        context_left = tf.nn.relu(tf.nn.bias_add(left_h, self.b), "relu")  # TODO 这里和mode2文件一样，和model文件不一样
        return context_left
        # shape:[batch_size,embed_size]

    def get_context_right(self, context_right, embedding_afterward):
        """
        :param context_right:
        :param embedding_afterward:
        :return: output:[None,embed_size]
        """
        right_c = tf.matmul(context_right, self.W_r)
        # shape:[batch_size,embed_size]<---------context_right:[batch_size,embed_size];W_r:[embed_size,embed_size]
        right_e = tf.matmul(embedding_afterward, self.W_sr)
        # shape:[batch_size,embed_size]<----------embedding_afterward:[batch_size,embed_size];W_sr:[embed_size,embed_size]
        right_h = right_c + right_e
        # shape:[batch_size,embed_size]
        # context_right=self.activation(right_h) #shape:[batch_size,embed_size] #TODO
        context_right = tf.nn.relu(tf.nn.bias_add(right_h, self.b), "relu")  # TODO 这里和mode2文件一样，和model文件不一样
        return context_right
        # shape:[batch_size,embed_size]

    def conv_layer_with_recurrent_structure(self):
        """
        input:self.embedded_words:[None,sentence_length,embed_size]
        :return: shape:[None,sentence_length,embed_size*3]
        """
        # 1. get splitted list of word embeddings
        embedded_words_split = tf.split(self.User_Matrix, self.sequence_length,
                                        axis=1)  # sentence_length个[None,1,embed_size]
        embedded_words_squeezed = [tf.squeeze(x, axis=1) for x in embedded_words_split]
        # sentence_length个[None,embed_size]
        embedding_previous = self.left_side_first_word
        # tf.zeros((self.batch_size,self.embed_size)) #TODO SHOULD WE ASSIGN A VARIABLE HERE
        context_left_previous = tf.zeros((self.batch_size, self.embed_size))
        # self.left_side_context_first# TODO SHOULD WE ASSIGN A VARIABLE HERE

        # 2. get list of context left
        context_left_list = []
        for i, current_embedding_word in enumerate(embedded_words_squeezed):  # sentence_length个[None,embed_size]
            context_left = self.get_context_left(context_left_previous, embedding_previous)
            # [None,embed_size]
            context_left_list.append(context_left)  # append result to list
            embedding_previous = current_embedding_word  # assign embedding_previous
            context_left_previous = context_left  # assign context_left_previous

        # 3. get context right
        embedded_words_squeezed2 = copy.copy(embedded_words_squeezed)
        embedded_words_squeezed2.reverse()
        embedding_afterward = self.right_side_last_word  # tf.zeros((self.batch_size,self.embed_size)) # TODO self.right_side_last_word SHOULD WE ASSIGN A VARIABLE HERE
        context_right_afterward = tf.zeros(
            (self.batch_size, self.embed_size))  # self.right_side_context_last # TODO SHOULD WE ASSIGN A VARIABLE HERE
        context_right_list = []
        for j, current_embedding_word in enumerate(embedded_words_squeezed2):
            context_right = self.get_context_right(context_right_afterward, embedding_afterward)
            context_right_list.append(context_right)
            embedding_afterward = current_embedding_word
            context_right_afterward = context_right

        # 4.ensemble "left,embedding,right" to output
        output_list = []
        for index, current_embedding_word in enumerate(embedded_words_squeezed):
            # print('current_embedding_word.shape:',current_embedding_word.shape)
            # print('context_right_list[index]:',context_right_list[index].shape)
            representation = tf.concat([context_left_list[index], current_embedding_word, context_right_list[index]],
                                       axis=1)
            # representation's shape:[None,embed_size*3]
            output_list.append(representation)  # shape:sentence_length个[None,embed_size*3]

        # 5. stack list to a tensor
        output = tf.stack(output_list, axis=1)  # shape:[None,sentence_length,embed_size*3]
        return output

    def attention_weights_start(self, user_matrix, label_matrix):
        label_matrix = tf.transpose(label_matrix)
        # Shape: (embed_size, num_labels)
        user_matrix = tf.reshape(user_matrix, [-1, self.embed_size])
        # Shape: (None * sentence_length, embed_size)
        attention_matrix = tf.matmul(user_matrix, label_matrix)
        # user_matrix: (None * sentence_length, embed_size)
        # label_matrix: (embed_size, num_labels)
        # Shape: (None * sentence_length, num_labels)
        attention_matrix = tf.reshape(attention_matrix, [-1, self.sequence_length, self.num_labels])
        # Shape: (None, sentence_length, num_labels)
        user_matrix = tf.reshape(user_matrix, [-1, self.sequence_length, self.embed_size])
        # Shape: (None, sentence_length, embed_size)
        word_weights = tf.nn.top_k(attention_matrix, k=1, sorted=False)[0]
        # Shape: (None, sentence_length, 1)
        attention_matrix_transpose = tf.transpose(attention_matrix, perm=[0, 2, 1])
        # Shape: (None, num_labels, sentence_length)
        label_weights = tf.nn.top_k(attention_matrix_transpose, k=1, sorted=False)[0]
        # Shape: (None, num_labels, 1)
        user_matrix = tf.multiply(user_matrix, word_weights)
        # Shape: (None, sentence_length, embed_size)
        label_matrix = tf.transpose(label_matrix)
        # Shape: (num_labels, embed_size)
        label_matrix = tf.multiply(label_matrix, label_weights)
        # Shape: (None, num_labels, embed_size)
        return user_matrix, label_matrix

    def attention_weights_afterward(self, user_matrix, label_matrix):
        label_matrix = tf.transpose(label_matrix, perm=[0, 2, 1])
        # Shape: (None, embed_size, num_labels)
        attention_matrix = tf.matmul(user_matrix, label_matrix)
        # user_matrix: (None, sentence_length, embed_size)
        # label_matrix: (None, embed_size, num_labels)
        # Shape: (None, sentence_length, num_labels)
        word_weights = tf.nn.top_k(attention_matrix, k=1, sorted=False)[0]
        # Shape: (None, sentence_length, 1)
        attention_matrix_transpose = tf.transpose(attention_matrix, perm=[0, 2, 1])
        # Shape: (None, num_labels, sentence_length)
        label_weights = tf.nn.top_k(attention_matrix_transpose, k=1, sorted=False)[0]
        # Shape: (None, num_ labels, 1)
        user_matrix = tf.multiply(user_matrix, word_weights)
        # Shape: (None, sentence_length, embed_size)
        label_matrix = tf.transpose(label_matrix, perm=[0, 2, 1])
        # Shape: (None, num_labels, embed_size)
        label_matrix = tf.multiply(label_matrix, label_weights)
        # Shape: (None, num_labels, embed_size)
        return user_matrix, label_matrix

    def inference(self):
        self.User_Matrix = tf.nn.embedding_lookup(self.User_Embedding, self.input_x)
        # Shape: (None, sentence_length, embed_size)  保存用户动态的embedding矩阵
        self.User_Matrix, self.Label_Matrix = self.attention_weights_start(self.User_Matrix, self.Label_Matrix)
        # User_Matrix: (None, sentence_length, embed_size)
        # Label_Matrix: (None, num_labels, embed_size)
        for i in range(self.attention_loops):
            self.User_Matrix, self.Label_Matrix = self.attention_weights_afterward(self.User_Matrix, self.Label_Matrix)
        # User Matrix: (None, sentence_length, embed_size)
        # Label Matrix: (None, num_labels, embed_size)  保存各label的embedding矩阵

        output_conv = self.conv_layer_with_recurrent_structure()
        self.sentence_embeddings_expanded = tf.expand_dims(output_conv, -1)
        print("use single layer CNN")
        h = self.cnn_single_layer()
        # Shape: (None, self.num_filters_total)
        dense_result = tf.matmul(h, self.W_projection) + self.b_projection
        # (None, self.embed_size)
        #  tf.matmul()用于返回两个矩阵乘积的结果    全连接（FC）层
        result = tf.expand_dims(input=dense_result, axis=2)
        # Shape: (None, embed_size, 1)
        logits = tf.matmul(self.Label_Matrix, result)
        # Label_Matrix: (None, num_labels, embed_size)
        # result: (None, embed_size, 1)
        # logits: (None, num_labels, 1)
        logits = tf.squeeze(logits, [2])
        # Shape: (None, num_labels)
        return logits

    def cnn_single_layer(self):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):  # enumerate列出数据和数据下标
            # 例如filter_sizes=[6,7,8]时，每次循环遍历filter_size依次等于6，7，8
            # with tf.name_scope("convolution-pooling-%s" %filter_size):
            with tf.variable_scope("convolution-pooling-%s" % filter_size):
                # tf.variable_scope可以让tf.Variable和tf.get_variable的变量具有相同的名字
                # ====>a.create filter
                filter = tf.get_variable("filter-%s" % filter_size,
                                         [filter_size, self.embed_size * 3, 1, self.num_filters],
                                         initializer=self.initializer)
                # 卷积核filter的维度应该是[filter_height, filter_width, in_channels, num_filters(out_channels)]
                # ====>b.conv operation: conv2d===>computes a 2-D convolution given 4-D `input` and `filter` tensors.
                # Conv.Input: given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`
                # Conv.Returns: A `Tensor`. Has the same type as `input`.
                #         A 4-D tensor. The dimension order is determined by the value of `data_format`, see below for details.
                # 1)each filter with conv2d's output a shape:[1,sequence_length-filter_size+1,1,1];2)*num_filters--->[1,sequence_length-filter_size+1,1,num_filters];3)*batch_size--->[batch_size,sequence_length-filter_size+1,1,num_filters]
                # input data format:NHWC:[batch, height, width, channels];output:4-D
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                # # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
                # 第一个参数input的维度是[batch, in_height, in_width, in_channels]
                # output的维度是[batch,filter_height,filter_width,num_filters]
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn_bn_')
                # tf.contrib.layers.batch_norm()用来进行批量归一化,具有统一规格的数据, 能让机器学习更容易学习到数据之中的规律

                # ====>c. apply nolinearity
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv, b),
                               "relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                #  tf.nn.relu计算激活函数relu，将矩阵的非最大值置零;tf.nn.bias_add(value,bias,name=None)将偏差项bias加到value上面
                # 激活函数使神经网络的输出成为一个非线性函数，使y的输出更复杂，激活函数增强了神经网络的表达能力
                # ====>. max-pooling.  value: A 4-D `Tensor` with shape `[batch, height, width, channels]
                #                  ksize: A list of ints that has length >= 4.  The size of the window for each dimension of the input tensor.
                #                  strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                # shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                # tf.nn.max_pool(value, ksize, strides, padding, name=None)
                # value为池化层的输入，由于池化层通常接在卷积层的后面，input通常为feature map，shape是[batch, height, width, channels]
                # ksize是池化窗口的大小，如[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
                # strides:和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
                # output:shape仍是[batch, height, width, channels]
                # print('i,pooled:',i,pooled.shape)
                pooled_outputs.append(pooled)
                #  把所有max_pooling层的输出都append到一起
        # 3.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
        # e.g. >>> x1=tf.ones([3,3]);x2=tf.ones([3,3]);x=[x1,x2]
        #         x12_0=tf.concat(x,0)---->x12_0' shape:[6,3]
        #         x12_1=tf.concat(x,1)---->x12_1' shape;[3,6]
        self.h_pool = tf.concat(pooled_outputs, 3)
        # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        #  tf.concat()函数用于在指定维度上进行张量的拼接
        # print('h_pool:',self.h_pool.shape)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])
        # shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)
        # tf.reshape(tensor,shape,name=None)作用是将tensor变换为参数shape形式

        # 4.=====>add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)  # [None,num_filters_total]
        h = tf.layers.dense(self.h_drop, self.num_filters_total, activation=tf.nn.tanh, use_bias=True)
        # 全连接层用来把所有的局部特征结合变成全局特征，用来计算最后每一类的得分；
        # 函数的前两个参数分别是该层的输入和输入的大小（维度）
        return h

    def cnn_multiple_layers(self):
        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        print("sentence_embeddings_expanded:", self.sentence_embeddings_expanded)
        for i, filter_size in enumerate(self.filter_sizes):
            # 例如filter_sizes=[6,7,8]时，每次循环遍历filter_size依次等于6，7，8
            with tf.variable_scope('cnn_multiple_layers' + "convolution-pooling-%s" % filter_size):
                # 1) CNN->BN->relu
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],
                                         initializer=self.initializer)
                # 卷积核filter的维度应该是[filter_height, filter_width, in_channels, num_filters(out_channels)]
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="SAME",
                                    name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
                # 第一个参数input的维度是[batch, in_height, in_width, in_channels]
                # output的维度是[batch,filter_height,filter_width,num_filters]
                # # padding参数为‘SAME’时，表示卷积核可以停留在图像边缘，可选参数有'SAME'和’VALID'
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn1')
                # 批标准化：具有统一规格的数据, 能让机器学习更容易学习到数据之中的规律
                print(i, "conv1:", conv)
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv, b),
                               "relu")  # shape:[batch_size,sequence_length,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                # 2) CNN->BN->relu
                # 第一层卷积核的输出作为第二层卷积核的输入
                h = tf.reshape(h, [-1, self.sequence_length, self.num_filters, 1])
                # shape:[batch_size,sequence_length,num_filters,1]，h的维度应该和self.sentence_embeddings_expanded的维度相同，以作为卷积层的输入
                # Layer2:CONV-RELU
                filter2 = tf.get_variable("filter2-%s" % filter_size,
                                          [filter_size, self.num_filters, 1, self.num_filters],
                                          initializer=self.initializer)
                conv2 = tf.nn.conv2d(h, filter2, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
                # shape:[batch_size,sequence_length-filter_size*2+2,1,num_filters]
                # padding参数为‘SAME’时，表示卷积核可以停留在图像边缘，可选参数有'SAME'和’VALID‘
                conv2 = tf.contrib.layers.batch_norm(conv2, is_training=self.is_training_flag, scope='cnn2')
                print(i, "conv2:", conv2)
                b2 = tf.get_variable("b2-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv2, b2),
                               "relu2")  # shape:[batch_size,sequence_length,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                # 3. Max-pooling
                pooling_max = tf.squeeze(tf.nn.max_pool(h, ksize=[1, self.sequence_length, 1, 1],
                                                        strides=[1, 1, 1, 1], padding='VALID', name="pool"))
                # tf.nn.max_pool(value, ksize, strides, padding, name=None)
                # value为池化层的输入，由于池化层通常接在卷积层的后面，input通常为feature map，shape是[batch, height, width, (out_)channels]
                # ksize是池化窗口的大小，如[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
                # strides:和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
                # output:shape仍是[batch, height, width, (out_)channels]
                # tf.squeeze()函数返回一个张量，这个张量是将原始input中所有维度为1的那些维都删掉的结果，axis可以用来指定要删掉的为1的维度，此处指定的维度必须确保其是1，否则会报错
                # pooling_avg=tf.squeeze(tf.reduce_mean(h,axis=1)) #[batch_size,num_filters]
                print(i, "pooling:", pooling_max)
                # pooling=tf.concat([pooling_max,pooling_avg],axis=1) #[batch_size,num_filters*2]
                pooled_outputs.append(pooling_max)  # h:[batch_size,sequence_length,1,num_filters]
        # concat
        h = tf.concat(pooled_outputs, axis=1)  # [batch_size,num_filters*len(self.filter_sizes)]
        # tf.concat()函数用于在指定维度上进行张量的拼接
        print("h.concat:", h)

        with tf.name_scope("dropout"):
            h = tf.nn.dropout(h,
                              keep_prob=self.dropout_keep_prob)  # [batch_size,sequence_length - filter_size + 1,num_filters]
        return h  # [batch_size,sequence_length - filter_size + 1,num_filters]

    def loss_multilabel(self, l2_lambda=0.0001):  # 0.0001#this loss function is for multi-label classification
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)
            losses = tf.reduce_sum(losses, axis=1)
            # shape=(?,). loss for all data in the batch
            loss = tf.reduce_mean(losses)
            # shape=().   average loss in the batch
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            # tf.nn.l2_loss一般用于优化的目标函数中的正则项，防止参数过多造成过拟合
            loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        self.learning_rate_ = learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # tf.train.AdamOptimizer()是Adam优化算法：是一个寻找全局最优点的优化算法
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
        # optimizer.compute_gradients()函数对var_list中的变量计算loss的梯度，返回一个以元组(gradient, variable)组成的列表
        # 梯度修剪(Gradient Clipping)主要避免训练梯度爆炸和消失问题,让权重的更新限制在一个合适的范围
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        # tf.clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)
        # 通过权重梯度的总和的比率来截取多个张量的值。
        # t_list 是梯度张量， clip_norm 是截取的比率, 这个函数返回截取过的梯度张量和一个所有张量的全局范数。
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # ADD 2018.06.01
        # tf.get_collection()从一个集合中取出全部变量，是一个列表
        with tf.control_dependencies(update_ops):  # ADD 2018.06.01
            train_op = optimizer.apply_gradients(zip(gradients, variables))
            # control_dependencies(control_inputs)返回一个控制依赖的上下文管理器，
            # 使用with关键字可以让在这个上下文环境中的操作都在control_inputs 执行。
            # optimizer.apply_gradients()返回一个执行梯度更新的ops，也是进行梯度修剪，从而避免训练梯度爆炸和消失问题
        return train_op
