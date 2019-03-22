import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, args, Personal_Memory, Recipe_Embedding, Category_Embedding, General_Memory):
        self.learner = args.learner
        self.num_categories = args.num_categories
        self.num_users = args.num_users
        self.num_labels = args.num_labels
        self.embed_size = args.embed_size
        self.learning_rate = tf.Variable(args.lr, trainable=False, name='learning_rate')
        self.decay_steps = args.decay_steps
        self.decay_rate = args.decay_rate
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name='Epoch_Step')
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.high_level_score_coefficient = tf.constant(args.high_level_score_coefficient)
        # self.high_level_score_coefficient = tf.Variable(0.92, trainable=True, name='High_level_score_coefficient')
        self.Personal_Memory = Personal_Memory
        self.Recipe_Embedding = Recipe_Embedding
        self.Category_Embedding = Category_Embedding
        self.General_Memory = General_Memory
        self.beta_1 = args.beta_1
        self.beta_2 = args.beta_2
        self.alpha = args.alpha

        self.user_input = tf.placeholder(dtype=tf.int32,
                                         shape=[None, ],
                                         name='user_input')
        self.item_input = tf.placeholder(dtype=tf.int32, shape=[None, ], name='item_input')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, ], name='labels')
        # 这里self.labels的数据类型应该是tf.float32,计算self.loss()时用到tf.nn.sigmoid_cross_entropy_with_logits需要labels和logits的数据类型一致
        self.write_sign = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='write_sign')
        # 这里声明为float型是为了和self.beta相乘时的需要（类型一致）
        self.categories = tf.placeholder(dtype=tf.float32, shape=[None, self.num_categories, 1], name='categories')
        self.user_one_hot_label = tf.placeholder(dtype=tf.float32, shape=[None, self.num_labels], name='user_labels')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.is_training_flag = tf.placeholder(tf.bool, name='is_training_flag')

        self.instantiate_weights()
        self.logits = self.inference()
        self.loss_value = self.loss()
        self.train_op = self.train()

    def instantiate_weights(self):
        with tf.name_scope('Embedding'):  # todo : 这句话有用吗
            self.Personal_Memory = tf.Variable(self.Personal_Memory)
            # Shape: (num_users, 5, len(user_feature))  按序保存了用户及其Memory Matrix的内容
            # len(user_feature) = 200
            self.Recipe_Embedding = tf.Variable(self.Recipe_Embedding)
            # Shape (num_items, len(dish_feature))  按序保存了菜谱及其Embedding Matrix的内容
            # len(dish_feature) = 200
            self.Category_Embedding = tf.Variable(self.Category_Embedding)
            # Shape: (num_categories, len(dish_feature))
            self.General_Memory = tf.Variable(self.General_Memory)
            # Shape: (num_labels, 5, len(user_feature))

    def inference(self):
        self.User_Memory = tf.nn.embedding_lookup(self.Personal_Memory, self.user_input)
        # Shape: (None, 5, 200)  指用户的整个Memory矩阵
        self.User_High_Level_Memory, self.User_Low_Level_Memory = tf.split(value=self.User_Memory,
                                                                           num_or_size_splits=[1, 4], axis=1)
        # self.User_High_Level_Memory: (None, 1, 200)
        # self.User_Low_Level_Memory: (None, 4, 200)
        # 分别指用户的General Memory和各个类别菜谱的Memory（先High Level,再Low Level）
        self.Item_Embedding = tf.nn.embedding_lookup(self.Recipe_Embedding, self.item_input)
        # Shape: (None, 200)  指当前batch中所有菜谱对应的embedding
        self.Item_Embedding = tf.expand_dims(input=self.Item_Embedding, axis=1)
        # Shape: (None, 1, 200)
        self.Dish_Category = tf.multiply(self.categories, self.Category_Embedding)
        # self.categories: (None, 4, 1)
        # self.Category_Embedding: (4, 200)
        # self.Dish_Category: (None, 4, 200)  指当前batch所有菜谱各自对应的category embedding，如果不属于某一类，则对应值为零
        self.category_score = tf.multiply(self.User_High_Level_Memory, self.Dish_Category)
        # self.User_High_Level_Memory: (None, 1, 200)
        # self.Dish_Category: (None, 4, 200)
        # self.category_score: (None, 4, 200)  指用户的High Level Memory和菜谱category embedding相乘后的结果，这里的相乘是对应元素相乘
        self.reduce_sum_category_score = tf.reduce_sum(input_tensor=self.category_score, axis=[1, 2])
        # self.reduce_sum_score: (None, )  指用户的High Level Memory和菜谱category embedding相乘后的score（多类别加到一起）
        category_num = tf.reduce_sum(self.categories, axis=[1, 2])
        # Shape: (None, )  指每道菜谱分别属于多少类
        self.high_score = tf.div(self.reduce_sum_category_score, category_num)
        # Shape: (None, )  所类别的情况取平均，得到最后的high level score

        self.Dish_Memory = tf.multiply(self.categories, self.User_Low_Level_Memory)
        # self.categories: (None, 4, 1)
        # self.User_Low_Level_Memory: (None, 4, 200)
        # self.Dish_Memory: (None, 4, 200)  指batch中的每道菜根据类别对应的User Dish Memory的结果（不属于某一类则该类值为零）
        self.dish_score = tf.multiply(self.Item_Embedding, self.Dish_Memory)
        # self.Item_Embedding: (None, 1, 200)
        # self.Dish_Memory: (None, 4, 200)
        # self.dish_score: (None, 4, 200)  指菜谱自身的item embedding和user memory相乘后的结果（对应元素相乘）
        self.reduce_sum_dish_score = tf.reduce_sum(input_tensor=self.dish_score, axis=[1, 2])
        # self.reduce_sum_dish_score: (None, )  指菜谱自身的item embedding和user memory相乘后的score（多类别加到一起）
        self.low_score = tf.div(self.reduce_sum_dish_score, category_num)
        # self.low_score: (None, )  最终的low level score

        # self.high_score = tf.nn.sigmoid(self.high_score)
        # self.low_score = tf.nn.sigmoid(self.low_score)
        # self.high_score = tf.minimum(tf.maximum(self.high_score, 0.0001), 1.0 - 0.0001)
        # self.low_score = tf.minimum(tf.maximum(self.low_score, 0.0001), 1.0 - 0.0001)

        self.score = self.high_level_score_coefficient * self.high_score + \
                        (1 - self.high_level_score_coefficient) * self.low_score
        # self.score = self.high_level_score_coefficient * self.high_score + \
        #              (tf.Variable(1.0) - self.high_level_score_coefficient) * self.low_score
        return self.score

    def loss(self):
        with tf.name_scope("loss"):
            # self.cross_entropy_losses = - tf.multiply(self.labels, tf.log(self.logits)) \
            #                             - tf.multiply((1 - self.labels), tf.log(1 - self.logits))
            self.cross_entropy_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            # Shape: (None,)
            loss = tf.reduce_mean(self.cross_entropy_losses)
            # tf.reduce_mean函数把所有的元素都加起来求平均值
        return loss

    def Write_Memory(self):
        # 下面三段代码用来更新Personal Memory
        item_embedding = tf.nn.embedding_lookup(self.Recipe_Embedding, self.item_input)
        # Shape: (None, 200)  指当前batch中所有菜谱对应的embedding
        item_embedding = tf.expand_dims(item_embedding, [1])
        # Shape: (None, 1, 200)
        dish_memory = tf.multiply(self.categories, item_embedding)
        # self.categories: (None, 4, 1)
        # item_embedding: (None, 1, 200)
        # dish_memory: (None, 4, 200)
        # 指用来准备和User_Low_Level_Memory做运算的dish embedding矩阵，它和User_Low_Level_Memory同等维度，当这道菜属于某一类时，
        # 这个维度上保存了这道菜的dish feature,否则这一维度上的数值为零
        low_coefficient = self.beta_1 * self.write_sign
        # Shape: (None, 1)
        low_coefficient = tf.expand_dims(low_coefficient, 1)
        # Shape: (None, 1, 1)
        dish_memory = tf.multiply(dish_memory, low_coefficient)
        # dish_memory: (None, 4, 200)
        # low_coefficient: (None, 1, 1)
        # dish_memory: (None, 4, 200)  这时dish_memory保存了更新时需向low level memory中相加的部分

        dish_category = tf.multiply(self.categories, self.Category_Embedding)
        # self.categories: (None, 4, 1)
        # self.Category_Embedding: (4, 200)
        # dish_category: (None, 4, 200)  指当前batch所有菜谱各自对应的category embedding，如果不属于某一类，则对应值为零
        dish_category = tf.reduce_sum(dish_category, axis=[1])
        # Shape: (None, 200)
        category_num = tf.reduce_sum(self.categories, axis=[1, 2])
        # Shape: (None, )
        dish_category = tf.div(dish_category, category_num)
        # Shape: (None, 200)
        # dish_category = tf.reduce_mean(dish_category, 1)
        # # Shape: (None, 200)  当前batch所有菜谱各自对应的Category Embedding的平均值
        dish_category = tf.expand_dims(dish_category, 1)
        # Shape: (None, 1, 200)
        high_coefficient = self.beta_2 * self.write_sign
        # Shape: (None, 1)
        high_coefficient = tf.expand_dims(high_coefficient, 1)
        # Shape: (None, 1, 1)
        dish_category = tf.multiply(dish_category, high_coefficient)
        # dish_category: (None, 1, 200)
        # high_coefficient: (None, 1, 1)
        # dish_category: (None, 1, 200)  dish_category保存了需要向high level memory中相加的部分

        dish_memory = tf.expand_dims(tf.reshape(dish_memory, [-1, self.num_categories * self.embed_size]), 1)
        # dish_memory: (None, 1, 800)
        user_input_onehot = tf.expand_dims(tf.one_hot(self.user_input, depth=self.num_users), 2)
        # user_input: (None, )
        # user_input_onehot: (None, num_user, 1)
        dish_bias = tf.reduce_sum(tf.matmul(user_input_onehot, dish_memory), 0)
        # (num_user, 800)
        dish_bias = tf.reshape(dish_bias, [-1, self.num_categories, self.embed_size])
        # (num_user, 4, 200)
        category_bias = tf.reduce_sum(tf.matmul(user_input_onehot, dish_category), 0)
        # user_input_onehot: (None, num_user, 1)
        # dish_category: (None, 1, 200)
        # Shape: (num_user, 200)
        category_bias = tf.expand_dims(category_bias, 1)
        # Shape: (num_user, 1, 200)
        # bias = tf.concat([dish_bias, category_bias], 1)
        bias = tf.concat([category_bias, dish_bias], 1)
        # Shape: (num_user, 5, 200)  (concat时先category,再dish)
        self.Personal_Memory = tf.assign(self.Personal_Memory, self.Personal_Memory + bias)
        # Shape: (num_user, 5, 200)

        # 下面一段代码用General Memory更新Personal Memory
        user_label_onehot = tf.expand_dims(self.user_one_hot_label, 2)
        # Shape: (None, num_labels, 1)
        general_memory = self.General_Memory
        # Shape: (num_labels, 5, 200)
        general_memory = tf.reshape(general_memory, [-1, (self.num_categories + 1) * self.embed_size])
        # Shape: (num_labels, 1000)
        user_label_memory = tf.multiply(user_label_onehot, general_memory)
        # user_label_onehot: (None, num_labels, 1)
        # general_memory: (num_labels, 1000)
        # user_label_memory: (None, num_labels, 1000)  指每个user根据其label对应的General Memory矩阵，如果不属于某一label，则对应值为零
        user_label_memory = tf.reduce_sum(user_label_memory, axis=1)
        # Shape: (None, 1000)
        user_num_labels = tf.reduce_sum(self.user_one_hot_label, axis=1)
        # Shape: (None, )  保存每个用户具有多少label
        user_label_memory = tf.div(user_label_memory, user_num_labels)
        # Shape: (None, 1000)  对用户的General Memory取平均得到的结果
        user_label_memory = tf.expand_dims(user_label_memory, 1)
        # Shape: (None, 1, 1000)
        general_bias = tf.reduce_sum(tf.matmul(user_input_onehot, user_label_memory), 0)
        # user_input_onehot: (None, num_users, 1)
        # user_label_memory: (None, 1, 1000)
        # general_bias: (num_users, 1000)
        general_bias = tf.reshape(general_bias, [-1, self.num_categories + 1, self.embed_size])
        # Shape: (num_users, 5, 200)
        self.Personal_Memory = tf.assign(self.Personal_Memory, self.Personal_Memory + general_bias)
        # Shape: (num_users, 5, 200)

        # 下面一段代码更新General Memory
        dish_general_bias = tf.reduce_sum(tf.matmul(user_label_onehot, dish_memory), 0)
        # user_label_onehot: (None, num_labels, 1)
        # dish_memory: (None, 1, 800)
        # Shape: (num_labels, 800)
        dish_general_bias = tf.reshape(dish_general_bias, [-1, self.num_categories, self.embed_size])
        # Shape: (num_labels, 4, 200)
        category_general_bias = tf.reduce_sum(tf.matmul(user_label_onehot, dish_category), 0)
        # user_label_onehot: (None, num_labels, 1)
        # dish_category: (None, 1, 200)
        # category_general_bias: (num_labels, 200)
        category_general_bias = tf.expand_dims(category_general_bias, 1)
        # Shape: (num_labels, 1, 200)
        general_bias = tf.concat([category_general_bias, dish_general_bias], 1)
        # Shape: (num_labels, 5, 200)
        self.General_Memory = tf.assign(self.General_Memory, self.General_Memory + general_bias)
        # Shape: (num_labels, 5, 200)

    def train(self):
        learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate, global_step=self.global_step,
                                                   decay_steps=self.decay_steps, decay_rate=self.decay_rate,
                                                   staircase=True)
        self.learning_rate = learning_rate
        if self.learner.lower() == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif self.learner.lower() == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
        elif self.learner.lower() == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_value))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        self.Write_Memory()
        return train_op
