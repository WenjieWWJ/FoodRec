import tensorflow as tf


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
        self.write_sign = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='write_sign')
        self.categories = tf.placeholder(dtype=tf.float32, shape=[None, self.num_categories, 1], name='categories')
        self.user_one_hot_label = tf.placeholder(dtype=tf.float32, shape=[None, self.num_labels], name='user_labels')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.is_training_flag = tf.placeholder(tf.bool, name='is_training_flag')

        self.instantiate_weights()
        self.logits = self.inference()
        self.loss_value = self.loss()
        self.personal, self.general = self.Write_Memory()
        self.train_op = self.train()

    def instantiate_weights(self):
        with tf.name_scope('Embedding'):
            self.Personal_Memory = tf.Variable(self.Personal_Memory)
            # Shape: (num_users, 5, len(user_feature))
            # len(user_feature) = 200
            self.Recipe_Embedding = tf.Variable(self.Recipe_Embedding)
            # Shape (num_items, len(dish_feature))
            # len(dish_feature) = 200
            self.Category_Embedding = tf.Variable(self.Category_Embedding)
            # Shape: (num_categories, len(dish_feature))
            self.General_Memory = tf.Variable(self.General_Memory)
            # Shape: (num_labels, 5, len(user_feature))

    def inference(self):
        self.User_Memory = tf.nn.embedding_lookup(self.Personal_Memory, self.user_input)
        # Shape: (None, 5, 200)
        self.User_High_Level_Memory, self.User_Low_Level_Memory = tf.split(value=self.User_Memory,
                                                                           num_or_size_splits=[1, 4], axis=1)
        # self.User_High_Level_Memory: (None, 1, 200)
        # self.User_Low_Level_Memory: (None, 4, 200)
        self.Item_Embedding = tf.nn.embedding_lookup(self.Recipe_Embedding, self.item_input)
        # Shape: (None, 200)
        self.Item_Embedding = tf.expand_dims(input=self.Item_Embedding, axis=1)
        # Shape: (None, 1, 200)
        self.Dish_Category = tf.multiply(self.categories, self.Category_Embedding)
        # self.categories: (None, 4, 1)
        # self.Category_Embedding: (4, 200)
        # self.Dish_Category: (None, 4, 200)
        self.category_score = tf.multiply(self.User_High_Level_Memory, self.Dish_Category)
        # self.User_High_Level_Memory: (None, 1, 200)
        # self.Dish_Category: (None, 4, 200)
        # self.category_score: (None, 4, 200)
        self.reduce_sum_category_score = tf.reduce_sum(input_tensor=self.category_score, axis=[1, 2])
        # self.reduce_sum_score: (None, )
        category_num = tf.reduce_sum(self.categories, axis=[1, 2])
        # Shape: (None, )
        self.high_score = tf.div(self.reduce_sum_category_score, category_num)
        # Shape: (None, )

        self.Dish_Memory = tf.multiply(self.categories, self.User_Low_Level_Memory)
        # self.categories: (None, 4, 1)
        # self.User_Low_Level_Memory: (None, 4, 200)
        # self.Dish_Memory: (None, 4, 200)
        self.dish_score = tf.multiply(self.Item_Embedding, self.Dish_Memory)
        # self.Item_Embedding: (None, 1, 200)
        # self.Dish_Memory: (None, 4, 200)
        # self.dish_score: (None, 4, 200)
        self.reduce_sum_dish_score = tf.reduce_sum(input_tensor=self.dish_score, axis=[1, 2])
        # self.reduce_sum_dish_score: (None, )
        self.low_score = tf.div(self.reduce_sum_dish_score, category_num)
        # self.low_score: (None, )

        self.score = self.high_level_score_coefficient * self.high_score + \
                        (1 - self.high_level_score_coefficient) * self.low_score
        return self.score

    def loss(self):
        with tf.name_scope("loss"):
            self.cross_entropy_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            # Shape: (None,)
            loss = tf.reduce_mean(self.cross_entropy_losses)
        return loss

    def Write_Memory(self):
        item_embedding = tf.nn.embedding_lookup(self.Recipe_Embedding, self.item_input)
        # Shape: (None, 200)
        item_embedding = tf.expand_dims(item_embedding, [1])
        # Shape: (None, 1, 200)
        dish_memory = tf.multiply(self.categories, item_embedding)
        # self.categories: (None, 4, 1)
        # item_embedding: (None, 1, 200)
        # dish_memory: (None, 4, 200)
        low_coefficient = self.beta_1 * self.write_sign
        # Shape: (None, 1)
        low_coefficient = tf.expand_dims(low_coefficient, 1)
        # Shape: (None, 1, 1)
        dish_memory = tf.multiply(dish_memory, low_coefficient)
        # dish_memory: (None, 4, 200)
        # low_coefficient: (None, 1, 1)
        # dish_memory: (None, 4, 200)

        dish_category = tf.multiply(self.categories, self.Category_Embedding)
        # self.categories: (None, 4, 1)
        # self.Category_Embedding: (4, 200)
        # dish_category: (None, 4, 200)
        dish_category = tf.reduce_sum(dish_category, axis=[1])
        # Shape: (None, 200)
        category_num = tf.reduce_sum(self.categories, axis=[1, 2])
        # Shape: (None, )
        category_num = tf.expand_dims(category_num, 1)
        # Shape: (None, 1)
        dish_category = tf.div(dish_category, category_num)
        # Shape: (None, 200)
        # dish_category = tf.reduce_mean(dish_category, 1)
        # # Shape: (None, 200)
        dish_category = tf.expand_dims(dish_category, 1)
        # Shape: (None, 1, 200)
        high_coefficient = self.beta_2 * self.write_sign
        # Shape: (None, 1)
        high_coefficient = tf.expand_dims(high_coefficient, 1)
        # Shape: (None, 1, 1)
        dish_category = tf.multiply(dish_category, high_coefficient)
        # dish_category: (None, 1, 200)
        # high_coefficient: (None, 1, 1)
        # dish_category: (None, 1, 200)

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
        # Shape: (num_user, 5, 200)
        self.Personal_Memory = tf.assign(self.Personal_Memory, self.Personal_Memory + bias)
        # Shape: (num_user, 5, 200)

        user_label_onehot = tf.expand_dims(self.user_one_hot_label, 2)
        # Shape: (None, num_labels, 1)
        general_memory = self.General_Memory
        # Shape: (num_labels, 5, 200)
        general_memory = tf.reshape(general_memory, [-1, (self.num_categories + 1) * self.embed_size])
        # Shape: (num_labels, 1000)
        user_label_memory = tf.multiply(user_label_onehot, general_memory)
        # user_label_onehot: (None, num_labels, 1)
        # general_memory: (num_labels, 1000)
        # user_label_memory: (None, num_labels, 1000)
        user_label_memory = tf.reduce_sum(user_label_memory, axis=1)
        # Shape: (None, 1000)
        user_num_labels = tf.reduce_sum(self.user_one_hot_label, axis=1)
        # Shape: (None, )
        user_num_labels = tf.expand_dims(user_num_labels, 1)
        # Shape: (None, 1)
        user_label_memory = tf.div(user_label_memory, user_num_labels)
        # Shape: (None, 1000)
        user_label_memory = tf.expand_dims(user_label_memory, 1)
        # Shape: (None, 1, 1000)
        general_bias = tf.reduce_sum(tf.matmul(user_input_onehot, user_label_memory), 0)
        # user_input_onehot: (None, num_users, 1)
        # user_label_memory: (None, 1, 1000)
        # general_bias: (num_users, 1000)
        general_bias = tf.reshape(general_bias, [-1, self.num_categories + 1, self.embed_size])
        # Shape: (num_users, 5, 200)
        general_bias = self.alpha * general_bias
        # Shape: (num_users, 5, 200)
        self.Personal_Memory = tf.assign(self.Personal_Memory, self.Personal_Memory + general_bias)
        # Shape: (num_users, 5, 200)

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

        personal = tf.reduce_mean(self.Personal_Memory)
        general = tf.reduce_mean(self.General_Memory)
        return personal, general


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
