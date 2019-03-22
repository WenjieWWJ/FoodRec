import tensorflow as tf
from Model_version1 import Model
import numpy as np
import pickle
import h5py
import os
import random
import heapq

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("cache_file_h5py", "Data/data.h5",
                       "path of training/validation/test data.")
# tf.flags.DEFINE_string("cache_file_h5py", "data.h5",
#                        "path of training/validation/test data.")
tf.flags.DEFINE_string("cache_file_pickle", "Data/vocab_label.pik",
                       "path of vocabulary and label files")
# tf.flags.DEFINE_string("cache_file_pickle", "vocab_label.pik",
#                        "path of vocabulary and label files.")
tf.flags.DEFINE_float("learning_rate", 0.0003, "learning rate")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.")
tf.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
tf.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")
tf.flags.DEFINE_string("ckpt_dir", "../../../share/jianghao/Text_model_checkpoint/Version2/loop_1/",
                       "checkpoint location for the model")
tf.flags.DEFINE_integer("sentence_len", 200, "max sentence length")
tf.flags.DEFINE_integer("embed_size", 128, "embedding size")
tf.flags.DEFINE_boolean("is_training_flag", True, "is training.true:tranining,false:testing/inference")
tf.flags.DEFINE_integer("num_epochs", 80, "number of epochs to run.")
tf.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.flags.DEFINE_integer("num_filters", 128, "number of filters")
tf.flags.DEFINE_boolean("multi_label_flag", True, "use multi label or single label.")
tf.flags.DEFINE_integer("num_labels", 96, "The number of user labels.")
tf.flags.DEFINE_integer("attention_loops", 1, "The number of attention loops.")
filter_sizes = [6, 7, 8]


def main(_):
    word2index, label2index, trainX, trainY, vaildX, vaildY, testX, testY = load_data(FLAGS.cache_file_h5py,
                                                                                      FLAGS.cache_file_pickle)
    vocab_size = len(word2index)
    print("cnn_model.vocab_size:", vocab_size)
    num_classes = len(label2index)
    print("num_classes:", num_classes)
    num_examples, FLAGS.sentence_len = trainX.shape  # trainX的两个维度分别表示num_examples和sentence_len
    print("num_examples of training:", num_examples, "sentence_len:", FLAGS.sentence_len)
    print("trainX[0]:", trainX[0])
    # train_X.shape:(60656, 200)
    print("trainY[0:10]:", trainY[0:10])
    # train_Y.shape:(60656, 96)
    train_y_short = get_target_label_short(trainY[0])
    # trainY[0]表示第一个样本数据的标签列表，维度是(96,)
    print("train_y_short:", train_y_short)
    # train_y_short是一个列表，列表的元素是label的index，比如trainY[0]在index=2,7,11处的标签值为1(其他为0),则train_y_short = [2,7,11]

    config = tf.ConfigProto()  # tf.ConfigProto()函数用在创建session的时候，用来对session进行参数配置
    config.gpu_options.allow_growth = True  # 让TensorFlow在运行过程中动态申请显存
    with tf.Session(config=config) as sess:  # 创建一个会话，会话用来管理TensorFlow程序运行时的所有资源
        # Instantiate Model;
        model = Model(FLAGS, filter_sizes, vocab_size)
        # Initialize Save
        saver = tf.train.Saver()  # 用于进行模型的保存
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            # for i in range(3): #decay learning rate if necessary.
            #    print(i,"Going to decay learning rate by half.")
            #    sess.run(textCNN.learning_rate_decay_half_op)
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())  # tf.global_variables_initializer()用来初始化模型的参数
        curr_epoch = sess.run(model.epoch_step)  # 通过会话tf.Session().run()进行循环优化网络参数

        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        iteration = 0
        validation_loss_so_far = 100
        best_F1_score = 0
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, counter = 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),
                                  range(batch_size, number_of_training_data, batch_size)):
                # (start,end)取值为两个range范围的取值相结合而形成的元组，每次取值递增batch_size个单位，end始终比start多一个batch_size(由range的初始值决定)
                iteration = iteration + 1
                if epoch == 0 and counter == 0:
                    print("trainX[start:end]:", trainX[start:end])
                feed_dict = {model.input_x: trainX[start:end], model.input_y_multilabel: trainY[start:end],
                             model.dropout_keep_prob: 0.8, model.is_training_flag: FLAGS.is_training_flag}
                curr_loss, lr, _ = sess.run([model.loss_val, model.learning_rate, model.train_op], feed_dict)
                loss, counter = loss + curr_loss, counter + 1
                if counter % 50 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tLearning rate:%.5f" % (
                        epoch, counter, loss / float(counter), lr))

                if start % (100 * FLAGS.batch_size) == 0:
                    eval_loss, f1_score, f1_micro, f1_macro, precision, recall = do_eval(sess, model, vaildX, vaildY,
                                                                                         num_classes)
                    print("Epoch %d Validation Loss:%.3f\t"
                          "F1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f\tPrecision:%.3f\tRecall:%.3f" %
                          (epoch, eval_loss, f1_score, f1_micro, f1_macro, precision, recall))
                    # save model to checkpoint
                    # save_path = FLAGS.ckpt_dir + "model.ckpt"
                    # print("Going to save model..")
                    # saver.save(sess, save_path, global_step=epoch)

            print("Going to increase epoch counter....")
            sess.run(model.epoch_increment)

            # validation
            if epoch % FLAGS.validate_every == 0:
                eval_loss, f1_score, f1_micro, f1_macro, precision, recall = do_eval(sess, model, vaildX, vaildY,
                                                                                     num_classes)
                print("Epoch %d Validation Loss:%.3f\t"
                      "F1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f\tPrecision:%.3f\tRecall:%.3f" %
                      (epoch, eval_loss, f1_score, f1_micro, f1_macro, precision, recall))
                if eval_loss < validation_loss_so_far or f1_score > best_F1_score:
                    validation_loss_so_far = eval_loss
                    best_F1_score = f1_score
                    subdirs = os.listdir(FLAGS.ckpt_dir)
                    for subdir in subdirs:
                        os.remove(FLAGS.ckpt_dir + subdir)
                        # 删除之前的eval_loss比当前大的检查点，防止占用过多空间
                    # save model to checkpoint
                    print('Going to save model and remove the checkpoint saved before...')
                    save_path = FLAGS.ckpt_dir + "model.ckpt"
                    saver.save(sess, save_path, global_step=epoch)

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss, f1_score, f1_micro, f1_macro, precision, recall = do_eval(sess, model, testX, testY, num_classes)
        print("Test Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f\tPrecision:%.3f\tRecall:%.3f" % (
            test_loss, f1_score, f1_micro, f1_macro, precision, recall))
    pass


# 在验证集上做验证，报告损失、精确度
def do_eval(sess, textCNN, evalX, evalY, num_classes):
    # evalX=evalX[0:2000]
    # evalY=evalY[0:2000]
    number_examples = len(evalX)
    eval_loss, eval_counter, eval_f1_score, eval_p, eval_r = 0.0, 0, 0.0, 0.0, 0.0
    batch_size = FLAGS.batch_size
    label_dict_confuse_matrix = init_label_dict(num_classes)  # init_label_dict()生成一个字典，给每一个类别三个值：TP,FP,FN
    # 混淆矩阵(confuse_matrix)每一列代表预测值，每一行代表的是实际的类别，通过混淆矩阵可以方便地看出预测结果哪里有错误，所有正确的预测结果都应该分布在对角线上
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        feed_dict = {textCNN.input_x: evalX[start:end], textCNN.input_y_multilabel: evalY[start:end],
                     textCNN.dropout_keep_prob: 1.0,
                     textCNN.is_training_flag: False}
        # curr_eval_acc--->textCNN.accuracy
        curr_eval_loss, logits = sess.run([textCNN.loss_val, textCNN.logits], feed_dict)
        for index, item in enumerate(evalY[start:end]):
            target_y = get_target_label_short(item)
            predict_y = get_label_using_logits(logits[index], target_y.__len__())
            # f1_score,p,r=compute_f1_score(list(label_list_top5), evalY[start:end][0])
            label_dict_confuse_matrix = compute_confuse_matrix(target_y, predict_y, label_dict_confuse_matrix)
            # compute_confuse_matrix()用来更新target_y,predict_y所涉及到的标签的TP,FP,FN的统计值
        eval_loss, eval_counter = eval_loss + curr_eval_loss, eval_counter + 1

    f1_micro, f1_macro, precision, recall = compute_micro_macro(label_dict_confuse_matrix)
    # label_dict_accusation is a dict, key is: accusation,value is: (TP,FP,FN). where TP is number of True Positive
    f1_score = (f1_micro + f1_macro) / 2.0
    return eval_loss / float(eval_counter), f1_score, f1_micro, f1_macro, precision, recall


#######################################
def compute_f1_score(predict_y, eval_y):
    """
    compoute f1_score.
    :param logits: [batch_size,label_size]
    :param evalY: [batch_size,label_size]
    :return:
    """
    f1_score = 0.0
    p_5 = 0.0
    r_5 = 0.0
    return f1_score, p_5, r_5


def compute_f1_score_removed(label_list_top5, eval_y):
    """
    compoute f1_score.
    :param logits: [batch_size,label_size]
    :param evalY: [batch_size,label_size]
    :return:
    """
    num_correct_label = 0
    eval_y_short = get_target_label_short(eval_y)
    for label_predict in label_list_top5:
        if label_predict in eval_y_short:
            num_correct_label = num_correct_label + 1
    # P@5 = Precision@5
    num_labels_predicted = len(label_list_top5)
    all_real_labels = len(eval_y_short)
    p_5 = num_correct_label / num_labels_predicted
    # R@5 = Recall@5
    r_5 = num_correct_label / all_real_labels
    f1_score = 2.0 * p_5 * r_5 / (p_5 + r_5 + 0.000001)
    return f1_score, p_5, r_5


random_number = 1000


def compute_confuse_matrix(target_y, predict_y, label_dict, name='default'):
    """
    compute true postive(TP,真阳性), false postive(FP,假阳性), false negative(FN,假阴性) given target lable and predict label
    :param target_y:
    :param predict_y:
    :param label_dict {label:(TP,FP,FN)}
    :return: macro_f1(a scalar),micro_f1(a scalar)
    """
    # 1.get target label and predict label
    if random.choice([x for x in range(random_number)]) == 1:  # random.choice()方法返回一个列表、元组或字符串的随机项
        print(name + ".target_y:", target_y, ";predict_y:", predict_y)  # debug purpose

    # 2.count number of TP,FP,FN for each class
    y_labels_unique = []
    y_labels_unique.extend(target_y)
    y_labels_unique.extend(predict_y)
    y_labels_unique = list(set(y_labels_unique))
    for i, label in enumerate(y_labels_unique):  # e.g. label=2
        TP, FP, FN = label_dict[label]
        if label in predict_y and label in target_y:  # predict=1,truth=1 (TP)
            TP = TP + 1
        elif label in predict_y and label not in target_y:  # predict=1,truth=0(FP)
            FP = FP + 1
        elif label not in predict_y and label in target_y:  # predict=0,truth=1(FN)
            FN = FN + 1
        label_dict[label] = (TP, FP, FN)
    return label_dict


def compute_micro_macro(label_dict):
    """
    compute f1 of micro and macro
    :param label_dict:
    :return: f1_micro,f1_macro: scalar, scalar
    """
    f1_micro, precision, recall = compute_f1_micro_use_TPFPFN(label_dict)
    f1_macro = compute_f1_macro_use_TPFPFN(label_dict)
    return f1_micro, f1_macro, precision, recall


def compute_TP_FP_FN_micro(label_dict):
    """
    compute micro FP,FP,FN
    :param label_dict_accusation: a dict. {label:(TP, FP, FN)}
    :return:TP_micro,FP_micro,FN_micro
    """
    TP_micro, FP_micro, FN_micro = 0.0, 0.0, 0.0
    for label, tuplee in label_dict.items():  # dict.items()函数以列表返回可遍历的(键, 值)元组数组
        TP, FP, FN = tuplee
        TP_micro = TP_micro + TP
        FP_micro = FP_micro + FP
        FN_micro = FN_micro + FN
    return TP_micro, FP_micro, FN_micro


def compute_f1_micro_use_TPFPFN(label_dict):
    """
    compute f1_micro
    :param label_dict: {label:(TP,FP,FN)}
    :return: f1_micro: a scalar(标量）
    """
    # 'micro':通过先计算总体的TP，FN和FP的数量，再计算F1
    TP_micro_accusation, FP_micro_accusation, FN_micro_accusation = compute_TP_FP_FN_micro(label_dict)
    # compute_TF_FP_FN_micro()函数的作用是计算所有label加起来的总TP,FN和FP的数量
    f1_micro_accusation, precision, recall = compute_f1(TP_micro_accusation, FP_micro_accusation, FN_micro_accusation,
                                                        'micro')
    # compute_f1()函数的作用是计算查准率，召回率和F1 score
    return f1_micro_accusation, precision, recall


def compute_f1_macro_use_TPFPFN(label_dict):
    """
    compute f1_macro
    :param label_dict: {label:(TP,FP,FN)}
    :return: f1_macro
    """
    # 'macro':分布计算每个类别的F1，然后做平均（各类别F1的权重相同）
    f1_dict = {}
    num_classes = len(label_dict)
    for label, tuplee in label_dict.items():
        TP, FP, FN = tuplee
        f1_score_onelabel, _, _ = compute_f1(TP, FP, FN, 'macro')
        f1_dict[label] = f1_score_onelabel  # 首先计算每一类的F1值
    f1_score_sum = 0.0
    for label, f1_score in f1_dict.items():
        f1_score_sum = f1_score_sum + f1_score
    f1_score = f1_score_sum / float(num_classes)  # 根据各类的F1值作和求平均
    return f1_score  # 得到F1_macro_score的值


small_value = 0.00001


def compute_f1(TP, FP, FN, compute_type):
    """
    compute f1
    :param TP_micro: number.e.g. 200
    :param FP_micro: number.e.g. 200
    :param FN_micro: number.e.g. 200
    :return: f1_score: a scalar
    """
    precison = TP / (TP + FP + small_value)  # 计算查准率
    recall = TP / (TP + FN + small_value)  # 计算召回率
    f1_score = (2 * precison * recall) / (precison + recall + small_value)  # 根据查准率和召回率，结合F1的计算公式计算F1的值

    # if random.choice([x for x in range(500)]) == 1:
    # print(compute_type,"precison:",str(precison),";recall:",str(recall),";f1_score:",f1_score)

    return f1_score, precison, recall


def init_label_dict(num_classes):
    """
    init label dict. this dict will be used to save TP,FP,FN
    :param num_classes:
    :return: label_dict: a dict. {label_index:(0,0,0)}
    """
    label_dict = {}
    for i in range(num_classes):
        label_dict[i] = (0, 0, 0)
    return label_dict


def get_target_label_short(eval_y):
    eval_y_short = []  # will be like:[2,7,11],表示在2,7,11处的标签值为1，其他处的标签值为0
    for index, label in enumerate(eval_y):  # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据的下标
        if label > 0:
            eval_y_short.append(index)
    return eval_y_short


# get top-N predicted labels
def get_label_using_logits(logits, length):
    # index_list=np.argsort(logits)[-top_number:]
    # vindex_list=index_list[::-1]

    # y_predict_labels = [i for i in range(len(logits)) if logits[i] >= 0.50]  # TODO 0.5PW e.g.[2,12,13,10]
    # if len(y_predict_labels) < 1: y_predict_labels = [np.argmax(logits)]  # np.argmax(logits)用于取出logits中元素最大值所对应的索引

    # print(logits)

    y_predict_labels = []
    copy = list(logits)
    for i in copy:
        if i > 0: y_predict_labels.append(copy.index(i))
    if y_predict_labels == []:
        y_predict_labels.append(np.argmax(copy))
    # print(y_predict_labels)

    return y_predict_labels


# 统计预测的准确率
def calculate_accuracy(labels_predicted, labels, eval_counter):
    label_nozero = []
    # print("labels:",labels)
    labels = list(labels)
    for index, label in enumerate(labels):
        if label > 0:
            label_nozero.append(index)
    if eval_counter < 2:
        print("labels_predicted:", labels_predicted, " ;labels_nozero:", label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
        if flag is not None:
            count = count + 1
    return count / len(labels)


def load_data(cache_file_h5py, cache_file_pickle):
    if not os.path.exists(cache_file_h5py) or not os.path.exists(cache_file_pickle):
        raise RuntimeError("############################ERROR##############################\n. "
                           "please download cache file, it include training data and vocabulary & labels. "
                           "link can be found in README.md\ndownload zip file, unzip it, then put cache files as FLAGS."
                           "cache_file_h5py and FLAGS.cache_file_pickle suggested location.")
    print("INFO. cache file exists. going to load cache file")
    f_data = h5py.File(cache_file_h5py, 'r')  # f_data作为相应h5文件的指代变量
    print("f_data.keys:", list(f_data.keys()))
    train_X = f_data['train_X']
    print("train_X.shape:", train_X.shape)
    train_Y = f_data['train_Y']
    print("train_Y.shape:", train_Y.shape, ";")
    vaild_X = f_data['vaild_X']
    valid_Y = f_data['valid_Y']
    test_X = f_data['test_X']
    test_Y = f_data['test_Y']
    # print(train_X)
    # f_data.close()

    word2index, label2index = None, None
    with open(cache_file_pickle, 'rb') as data_f_pickle:  # 文件打开方式'b'表示要对二进制数据进行读写
        word2index, label2index = pickle.load(
            data_f_pickle)  # pickle.dump()函数能一个接着一个地将几个对象转储到同一个文件;随后调用pickle.load()来以同样的顺序检索这些对象
    print("INFO. cache file load successful...")
    return word2index, label2index, train_X, train_Y, vaild_X, valid_Y, test_X, test_Y


if __name__ == "__main__":
    tf.app.run()
