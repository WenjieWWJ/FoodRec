import argparse
from time import time
from Dataset import Dataset
import tensorflow as tf
import os
from model_Recommender_Write_without_General import Model
from evaluate import evaluate_model
import numpy as np
import random
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Run Model.")
    # parser.add_argument('--path', nargs='?', default='Data2/',
    #                     help='Input data path.')
    # parser.add_argument('--path', nargs='?', default='../../../share/jianghao/Recommender_data/',
    #                     help='Input data path.')
    parser.add_argument('--path', nargs='?', default='../../../share/wangwenjie/jianghao/Recommender_data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--dish_to_category_file', nargs='?', default='dish_to_category.json',
                        help='Path of dish_to_category.json file. Saved the map from dish to category.')
    parser.add_argument('--user_to_one_hot_label_file', nargs='?', default='user_to_one_hot_label.json',
                        help='Path of user_to_one_hot_label.json file.')
    parser.add_argument('--label_to_user_file', nargs='?', default='label_to_user.json',
                        help='Path of label_to_user.json file.')
    parser.add_argument('--Personal_Memory_file', nargs='?', default='Personal_Memory.npy',
                        help='Path of Personal_Memory.npy file. Saved personal memory matrix.')
    parser.add_argument('--Recipe_Embedding_file', nargs='?', default='Recipe_Embedding.npy',
                        help='Path of Recipe_Embedding.npy file. Saved recipe embedding matrix.')
    parser.add_argument('--Category_Embedding_file', nargs='?', default='Category_Embedding.npy',
                        help='Path of Category_Embedding.npy file. Saved category embedding matrix.')
    parser.add_argument('--General_Memory_file', nargs='?', default='General_Memory.npy',
                        help='Path of General_Memory.npy file. Saved general memory matrix.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--ckpt_dir', nargs='?', default='checkpoint/',
                        help='Checkpoint path.')
    parser.add_argument('--decay_steps', type=int, default=1000,
                        help='how many steps before decay learning rate')
    parser.add_argument('--decay_rate', type=int, default=1.0,
                        help='Rate of decay for learning rate.')
    parser.add_argument('--num_dish_images', type=int, default=4548,
                        help='The number of dish images.')
    parser.add_argument('--num_users', type=int, default=64657,
                        help='The number of users.')
    # parser.add_argument('--num_dish_images', type=int, default=2,  # todo todo
    #                     help='The number of dish images.')
    # parser.add_argument('--num_users', type=int, default=2,  # todo todo
    #                     help='The number of users.')
    parser.add_argument('--num_categories', type=int, default=4,
                        help='The number of dish categories.')
    parser.add_argument('--num_labels', type=int, default=95,
                        help='The number of user labels.')
    parser.add_argument('--embed_size', type=int, default=200,
                        help='The shape of user feature and dish feature.')
    parser.add_argument('--high_level_score_coefficient', type=float, default=0.99,
                        help='The coefficient of high level score.')
    parser.add_argument('--low_level_score_coefficient', type=float, default=0.01,
                        help='The coefficient of low level score.')
    parser.add_argument('--beta_1', type=float, default=0.2,
                        help='The coefficient at low level when writing the memory matrix.')
    parser.add_argument('--beta_2', type=float, default=0.2,
                        help='The coefficient at high level when writing the memory matrix.')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='The coefficient used when add general memory to user memory.')
    return parser.parse_args()


def get_train_instances(train, testNegatives, dish_to_category, user_to_one_hot_label):
    # user_input_index, item_input_index中分别保存了训练集的user_input和item_input的index顺序序列，labels保存了对应的标签，
    # categories保存了这些菜对应的属于哪些类， write_sign列表对正例保存1，负例保存-1，目的是区分出正负例以确定Memory矩阵执行
    # 写入操作时的正负号
    user_input_index, item_input_index, labels, categories, write_sign, user_one_hot_label = [], [], [], [], [], []
    for user in train:
        sample_num = 200 if len(train[str(user)]) > 200 else len(train[str(user)])
        positive_instance = random.sample(train[str(user)], sample_num)
        for index, positive in enumerate(positive_instance):
            user_input_index.append(user)
            item_input_index.append(positive)
            categories.append(dish_to_category[str(positive)])
            labels.append(1)
            write_sign.append([1.0])
            user_one_hot_label.append(user_to_one_hot_label[str(user)])
        sample_num = 50 if len(testNegatives[str(user)]) > 50 else len(testNegatives[str(user)])
        negative_instance = testNegatives[str(user)][:sample_num]  # 前sample_num个负例用来训练
        for index, negative in enumerate(negative_instance):
            user_input_index.append(user)
            item_input_index.append(negative)
            categories.append(dish_to_category[str(negative)])
            labels.append(0)
            write_sign.append([-1.0])
            user_one_hot_label.append(user_to_one_hot_label[str(user)])
    #     data = []
    #     for index, positive in enumerate(positive_instance):
    #         sample = {}
    #         sample["user"] = user
    #         sample["type"] = positive
    #         sample["category"] = dish_to_category[str(positive)]
    #         sample["label"] = 1
    #         data.append(sample)
    #
    #     sample_num = 50 if len(testNegatives[str(user)]) > 50 else len(testNegatives[str(user)])
    #     negative_instance = testNegatives[str(user)][:sample_num]  # 前sample_num个负例用来训练
    #     for index, negative in enumerate(negative_instance):
    #         sample = {}
    #         sample["user"] = user
    #         sample["type"] = negative
    #         sample["category"] = dish_to_category[str(negative)]
    #         sample["label"] = 0
    #         data.append(sample)
    # random.shuffle(data)
    # for sample in data:
    #     user_input_index.append(sample["user"])
    #     item_input_index.append(sample["type"])
    #     categories.append(sample["category"])
    #     labels.append(sample["label"])

    return user_input_index, item_input_index, labels, categories, write_sign, user_one_hot_label


def load_numpy_file(path):
    array = np.load(path)
    return array


def load_json_file(path):
    dict = json.loads(open(path, 'r').read())
    return dict


if __name__ == '__main__':
    args = parse_args()

    topK = 10
    print("Model arguments: %s" % args)
    # model_out_dir = '../../../share/jianghao/Recommender/General_Memory/'
    model_out_dir = '../../../share/wangwenjie/jianghao/Recommender_checkpoint/Recommender_Write_without_General_0.2/'
    # checkpoint directory的命名方式是high level coefficient - low level coefficient

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 1.0  # 占用 GPU 90% 的显存
    # Create session
    with tf.Session(config=config) as sess:
        # Loading data
        t1 = time()
        dataset = Dataset(args.path + args.dataset)
        train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
        num_users, num_instances, num_test = args.num_users, dataset.num_instances, dataset.num_test

        Personal_Memory = load_numpy_file(args.path + args.Personal_Memory_file)
        Recipe_Embedding = load_numpy_file(args.path + args.Recipe_Embedding_file)
        Category_Embedding = load_numpy_file(args.path + args.Category_Embedding_file)
        General_Memory = load_numpy_file(args.path + args.General_Memory_file)
        dish_to_category = load_json_file(args.path + args.dish_to_category_file)
        user_to_one_hot_label = load_json_file(args.path + args.user_to_one_hot_label_file)
        num_items = args.num_dish_images
        print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
              % (time() - t1, num_users, num_items, num_instances, num_test))

        # Generate training instances
        user_input_index, item_input_index, labels, categories, write_sign, user_one_hot_label =\
            get_train_instances(train, testNegatives,dish_to_category, user_to_one_hot_label)
        number_of_training_data = len(user_input_index)
        print('Number of training data:', number_of_training_data)

        model = Model(args, Personal_Memory, Recipe_Embedding, Category_Embedding, General_Memory)
        saver = tf.train.Saver()
        if os.path.exists(model_out_dir + "checkpoint"):
            print("Restore variables from checkpoint.")
            saver.restore(sess, tf.train.latest_checkpoint(model_out_dir))
        else:
            print('Initialize variables.')
            sess.run(tf.global_variables_initializer())
        curr_epoch = sess.run(model.epoch_step)

        # Train model
        best_hr, best_ndcg, best_iter = 0, 0, -1  # todo todo todo todo
        for epoch in range(args.epochs):
            t1 = time()

            # Training
            loss = 0
            current_batch = 0
            for start, end in zip(range(0, number_of_training_data, args.batch_size),
                                  range(args.batch_size,
                                        int((number_of_training_data / args.batch_size) + 1) * args.batch_size,
                                        args.batch_size)):
                t3 = time()
                current_batch = current_batch + 1
                feed_dict = {model.user_input: user_input_index[start:end],
                             model.item_input: item_input_index[start:end],
                             model.labels: labels[start:end], model.categories: categories[start:end],
                             model.user_one_hot_label: user_one_hot_label[start:end],
                             model.write_sign: write_sign[start:end], model.dropout_keep_prob: 0.8,
                             model.is_training_flag: True}
                curr_loss, lr, high_score, low_score, score, high_coefficient, _ = sess.run(
                    [model.loss_value, model.learning_rate,
                     model.high_score, model.low_score, model.score,
                     model.high_level_score_coefficient, model.train_op],
                    feed_dict)
                loss = loss + curr_loss
                if current_batch % 100 == 0:
                    # print('high score:', high_score[:10])
                    # print('low score:', low_score[:10])
                    # print('labels:', labels[:10])
                    # print('score:', score[:10])
                    # print('high_level_coefficient:', high_coefficient)
                    print('batch % d : curr_loss = %.4f  [%.4f s]' %
                          (current_batch, curr_loss, time() - t3))
            sess.run(model.epoch_increment)
            t2 = time()

            # Evaluation
            if epoch % args.verbose == 0:
                (hits, ndcgs) = evaluate_model(sess, model, testRatings, testNegatives, topK, dish_to_category)
                hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), loss / current_batch
                print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f  [%.1f s]'
                      % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
                if hr > best_hr:
                    best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                    if args.out > 0:
                        print("Going to save model.")
                        subdirs = os.listdir(model_out_dir)
                        for subdir in subdirs:
                            os.remove(model_out_dir + subdir)
                        # save model to checkpoint
                        save_path = model_out_dir + "model.ckpt"
                        saver.save(sess, save_path, global_step=epoch)

            # Add test
            if epoch % args.verbose == 0:
                (hits, ndcgs) = evaluate_model(sess, model, testRatings, testNegatives, topK, dish_to_category)
                hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), loss / current_batch
                print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f  [%.1f s]'
                      % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))

        print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
        if args.out > 0:
            print("The best GMF model is saved to %s" % model_out_dir)
