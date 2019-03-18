import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, args, Personal_Memory, Recipe_Embedding, Category_Embedding):
        self.learner = args.learner
        self.learning_rate = tf.Variable(args.lr, trainable=False, name='learning_rate')
        self.decay_steps = args.decay_steps
        self.decay_rate = args.decay_rate
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name='Epoch_Step')
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.num_categories = args.num_categories
        self.high_level_score_coefficient = args.high_level_score_coefficient
        # self.high_level_score_coefficient = tf.Variable(0.92, trainable=True, name='High_level_score_coefficient')
        self.Personal_Memory = Personal_Memory
        self.Recipe_Embedding = Recipe_Embedding
        self.Category_Embedding = Category_Embedding
        # self.eps = tf.constant(0.0001)

        self.user_input = tf.placeholder(dtype=tf.int32,
                                         shape=[None, ],
                                         name='user_input')
        self.item_input = tf.placeholder(dtype=tf.int32, shape=[None, ], name='item_input')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, ], name='labels')
        # 这里self.labels的数据类型应该是tf.float32,计算self.loss()时用到tf.nn.sigmoid_cross_entropy_with_logits需要labels和logits的数据类型一致
        self.categories = tf.placeholder(dtype=tf.float32, shape=[None, self.num_categories, 1], name='categories')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.is_training_flag = tf.placeholder(tf.bool, name='is_training_flag')

        self.instantiate_weights()
        self.logits = self.inference()
        self.loss_value = self.loss()
        self.train_op = self.train()

    def instantiate_weights(self):
        with tf.name_scope('Embedding'):  # todo : 这句话有用吗
            self.Personal_Memory = tf.Variable(self.Personal_Memory)
            # Shape: (num_users, num_categories, len(user_feature))  按序保存了用户及其Memory Matrix的内容
            self.Recipe_Embedding = tf.Variable(self.Recipe_Embedding)
            # Shape (num_items, len(dish_feature))  按序保存了菜谱及其Embedding Matrix的内容
            self.Category_Embedding = tf.Variable(self.Category_Embedding)
            # Shape: (num_categories, len(dish_feature))

    def inference(self):
        self.User_Memory = tf.nn.embedding_lookup(self.Personal_Memory, self.user_input)
        # Shape: (None, 5, 200)  指用户的整个Memory矩阵
        self.User_High_Level_Memory, self.User_Low_Level_Memory = tf.split(value=self.User_Memory,
                                                                           num_or_size_splits=[1, 4], axis=1)
        # self.User_High_Level_Memory: (None, 1, 200)
        # self.User_Low_Level_Memory: (None, 4, 200)  分别指用户的General Memory和各个类别菜谱的Memory
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
        # self.category_score: (None, 4, 200)  指用户的General Memory和菜谱category embedding相乘后的结果，这里的相乘是对应元素相乘
        self.reduce_sum_category_score = tf.reduce_sum(input_tensor=self.category_score, axis=[1, 2])
        # self.reduce_sum_score: (None, )  指用户的General Memory和菜谱category embedding相乘后的score（多类别加到一起）
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
        return train_op
