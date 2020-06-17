import tensorflow as tf
import copy


class Model:
    def __init__(self, FLAGS, filter_sizes, vocab_size, clip_gradients=5.0):
        self.num_classes = FLAGS.num_labels
        self.batch_size = FLAGS.batch_size
        self.sequence_length = FLAGS.sentence_len
        self.vocab_size = vocab_size
        self.embed_size = FLAGS.embed_size
        self.learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False,
                                         name="learning_rate")
        self.filter_sizes = filter_sizes
        self.num_filters = FLAGS.num_filters
        self.num_filters_total = self.num_filters * len(filter_sizes)
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
                                                shape=[self.embed_size])

    def get_context_left(self, context_left, embedding_previous):
        left_c = tf.matmul(context_left, self.W_l)
        left_e = tf.matmul(embedding_previous, self.W_sl)
        left_h = left_c + left_e
        context_left = tf.nn.relu(tf.nn.bias_add(left_h, self.b), "relu")
        return context_left

    def get_context_right(self, context_right, embedding_afterward):
        right_c = tf.matmul(context_right, self.W_r)
        right_e = tf.matmul(embedding_afterward, self.W_sr)
        right_h = right_c + right_e
        context_right = tf.nn.relu(tf.nn.bias_add(right_h, self.b), "relu")
        return context_right

    def conv_layer_with_recurrent_structure(self):
        embedded_words_split = tf.split(self.User_Matrix, self.sequence_length,
                                        axis=1)
        embedded_words_squeezed = [tf.squeeze(x, axis=1) for x in embedded_words_split]
        embedding_previous = self.left_side_first_word
        context_left_previous = tf.zeros((self.batch_size, self.embed_size))

        context_left_list = []
        for i, current_embedding_word in enumerate(embedded_words_squeezed):
            context_left = self.get_context_left(context_left_previous, embedding_previous)
            context_left_list.append(context_left)
            embedding_previous = current_embedding_word
            context_left_previous = context_left

        # get context right
        embedded_words_squeezed2 = copy.copy(embedded_words_squeezed)
        embedded_words_squeezed2.reverse()
        embedding_afterward = self.right_side_last_word
        context_right_afterward = tf.zeros(
            (self.batch_size, self.embed_size))
        context_right_list = []
        for j, current_embedding_word in enumerate(embedded_words_squeezed2):
            context_right = self.get_context_right(context_right_afterward, embedding_afterward)
            context_right_list.append(context_right)
            embedding_afterward = current_embedding_word
            context_right_afterward = context_right

        output_list = []
        for index, current_embedding_word in enumerate(embedded_words_squeezed):
            representation = tf.concat([context_left_list[index], current_embedding_word, context_right_list[index]],
                                       axis=1)
            output_list.append(representation)

        output = tf.stack(output_list, axis=1)
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
        # Shape: (None, sentence_length, embed_size)
        self.User_Matrix, self.Label_Matrix = self.attention_weights_start(self.User_Matrix, self.Label_Matrix)
        # User_Matrix: (None, sentence_length, embed_size)
        # Label_Matrix: (None, num_labels, embed_size)
        for i in range(self.attention_loops):
            self.User_Matrix, self.Label_Matrix = self.attention_weights_afterward(self.User_Matrix, self.Label_Matrix)
        # User Matrix: (None, sentence_length, embed_size)
        # Label Matrix: (None, num_labels, embed_size)

        output_conv = self.conv_layer_with_recurrent_structure()
        self.sentence_embeddings_expanded = tf.expand_dims(output_conv, -1)
        print("use single layer CNN")
        h = self.cnn_single_layer()
        # Shape: (None, self.num_filters_total)
        dense_result = tf.matmul(h, self.W_projection) + self.b_projection
        # (None, self.embed_size)
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
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("convolution-pooling-%s" % filter_size):
                filter = tf.get_variable("filter-%s" % filter_size,
                                         [filter_size, self.embed_size * 3, 1, self.num_filters],
                                         initializer=self.initializer)
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn_bn_')

                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b),
                               "relu")
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)
        h = tf.layers.dense(self.h_drop, self.num_filters_total, activation=tf.nn.tanh, use_bias=True)
        return h

    def loss_multilabel(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)
            losses = tf.reduce_sum(losses, axis=1)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        self.learning_rate_ = learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op
