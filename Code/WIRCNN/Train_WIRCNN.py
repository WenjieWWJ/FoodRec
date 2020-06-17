import tensorflow as tf
from WIRCNN.Model_WIRCNN import Model
import numpy as np
import pickle
import h5py
import os


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("cache_file_h5py", "data.h5",
                       "path of training/validation/test data.")
tf.flags.DEFINE_string("cache_file_pickle", "vocab_label.pik",
                       "path of vocabulary and label files.")
tf.flags.DEFINE_float("learning_rate", 0.0003, "learning rate")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.")
tf.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
tf.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")
tf.flags.DEFINE_string("ckpt_dir", "checkpoint/",
                       "checkpoint location for the model")
tf.flags.DEFINE_integer("sentence_len", 390, "max sentence length")
tf.flags.DEFINE_integer("embed_size", 128, "embedding size")
tf.flags.DEFINE_boolean("is_training_flag", True,
                        "is training.true:tranining, false:testing/inference")
tf.flags.DEFINE_integer("num_epochs", 240, "number of epochs to run.")
tf.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.flags.DEFINE_integer("num_filters", 128, "number of filters")
tf.flags.DEFINE_boolean("multi_label_flag", True, "use multi label or single label.")
tf.flags.DEFINE_integer("num_labels", 96, "The number of user labels.")
tf.flags.DEFINE_integer("attention_loops", 0, "The number of attention loops.")
filter_sizes = [6, 7, 8]


def main(_):
    word2index, label2index, trainX, trainY, vaildX, vaildY, testX, testY = load_data(FLAGS.cache_file_h5py,
                                                                                      FLAGS.cache_file_pickle)
    vocab_size = len(word2index)
    num_classes = len(label2index)
    num_examples, FLAGS.sentence_len = trainX.shape

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = Model(FLAGS, filter_sizes, vocab_size)
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
        curr_epoch = sess.run(model.epoch_step)

        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        iteration = 0
        validation_loss_so_far = 100
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, counter = 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),
                                  range(batch_size, number_of_training_data, batch_size)):
                iteration = iteration + 1
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

            print("Going to increase epoch counter....")
            sess.run(model.epoch_increment)

            # validation
            if epoch % FLAGS.validate_every == 0:
                eval_loss, f1_score, f1_micro, f1_macro, precision, recall = do_eval(sess, model, vaildX, vaildY,
                                                                                     num_classes)
                print("Epoch %d Validation Loss:%.3f\t"
                      "F1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f\tPrecision:%.3f\tRecall:%.3f" %
                      (epoch, eval_loss, f1_score, f1_micro, f1_macro, precision, recall))
                if eval_loss < validation_loss_so_far:
                    validation_loss_so_far = eval_loss
                    # save model to checkpoint
                    print('Going to save model.')
                    save_path = FLAGS.ckpt_dir + "model.ckpt"
                    saver.save(sess, save_path, global_step=epoch)

        test_loss, f1_score, f1_micro, f1_macro, precision, recall = do_eval(sess, model, testX, testY, num_classes)
        print("Test Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f\tPrecision:%.3f\tRecall:%.3f" % (
            test_loss, f1_score, f1_micro, f1_macro, precision, recall))
    pass


def do_eval(sess, textCNN, evalX, evalY, num_classes):
    number_examples = len(evalX)
    eval_loss, eval_counter, eval_f1_score, eval_p, eval_r = 0.0, 0, 0.0, 0.0, 0.0
    batch_size = FLAGS.batch_size
    label_dict_confuse_matrix = init_label_dict(num_classes)
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        feed_dict = {textCNN.input_x: evalX[start:end], textCNN.input_y_multilabel: evalY[start:end],
                     textCNN.dropout_keep_prob: 1.0,
                     textCNN.is_training_flag: False}
        curr_eval_loss, logits = sess.run([textCNN.loss_val, textCNN.logits], feed_dict)
        for index, item in enumerate(evalY[start:end]):
            target_y = get_target_label_short(item)
            predict_y = get_label_using_logits(logits[index], target_y.__len__())
            label_dict_confuse_matrix = compute_confuse_matrix(target_y, predict_y, label_dict_confuse_matrix)
        eval_loss, eval_counter = eval_loss + curr_eval_loss, eval_counter + 1

    f1_micro, f1_macro, precision, recall = compute_micro_macro(label_dict_confuse_matrix)
    f1_score = (f1_micro + f1_macro) / 2.0
    return eval_loss / float(eval_counter), f1_score, f1_micro, f1_macro, precision, recall


def compute_confuse_matrix(target_y, predict_y, label_dict, name='default'):
    y_labels_unique = []
    y_labels_unique.extend(target_y)
    y_labels_unique.extend(predict_y)
    y_labels_unique = list(set(y_labels_unique))
    for i, label in enumerate(y_labels_unique):
        TP, FP, FN = label_dict[label]
        if label in predict_y and label in target_y:
            TP = TP + 1
        elif label in predict_y and label not in target_y:
            FP = FP + 1
        elif label not in predict_y and label in target_y:
            FN = FN + 1
        label_dict[label] = (TP, FP, FN)
    return label_dict


def compute_micro_macro(label_dict):
    f1_micro, precision, recall = compute_f1_micro_use_TPFPFN(label_dict)
    f1_macro = compute_f1_macro_use_TPFPFN(label_dict)
    return f1_micro, f1_macro, precision, recall


def compute_TP_FP_FN_micro(label_dict):
    TP_micro, FP_micro, FN_micro = 0.0, 0.0, 0.0
    for label, tuplee in label_dict.items():
        TP, FP, FN = tuplee
        TP_micro = TP_micro + TP
        FP_micro = FP_micro + FP
        FN_micro = FN_micro + FN
    return TP_micro, FP_micro, FN_micro


def compute_f1_micro_use_TPFPFN(label_dict):
    TP_micro_accusation, FP_micro_accusation, FN_micro_accusation = compute_TP_FP_FN_micro(label_dict)
    f1_micro_accusation, precision, recall = compute_f1(TP_micro_accusation, FP_micro_accusation, FN_micro_accusation,
                                                        'micro')
    return f1_micro_accusation, precision, recall


def compute_f1_macro_use_TPFPFN(label_dict):
    f1_dict = {}
    num_classes = len(label_dict)
    for label, tuplee in label_dict.items():
        TP, FP, FN = tuplee
        f1_score_onelabel, _, _ = compute_f1(TP, FP, FN, 'macro')
        f1_dict[label] = f1_score_onelabel
    f1_score_sum = 0.0
    for label, f1_score in f1_dict.items():
        f1_score_sum = f1_score_sum + f1_score
    f1_score = f1_score_sum / float(num_classes)
    return f1_score




def compute_f1(TP, FP, FN, compute_type):
    small_value = 0.00001
    precison = TP / (TP + FP + small_value)
    recall = TP / (TP + FN + small_value)
    f1_score = (2 * precison * recall) / (precison + recall + small_value)
    return f1_score, precison, recall


def init_label_dict(num_classes):
    label_dict = {}
    for i in range(num_classes):
        label_dict[i] = (0, 0, 0)
    return label_dict


def get_target_label_short(eval_y):
    eval_y_short = []
    for index, label in enumerate(eval_y):
        if label > 0:
            eval_y_short.append(index)
    return eval_y_short


def get_label_using_logits(logits, length):
    y_predict_labels = []
    copy = list(logits)
    for i in copy:
        if i > 0: y_predict_labels.append(copy.index(i))
    if y_predict_labels == []:
        y_predict_labels.append(np.argmax(copy))

    return y_predict_labels


def load_data(cache_file_h5py, cache_file_pickle):
    if not os.path.exists(cache_file_h5py) or not os.path.exists(cache_file_pickle):
        raise RuntimeError("No data file found.")
    f_data = h5py.File(cache_file_h5py, 'r')
    train_X = f_data['train_X']
    train_Y = f_data['train_Y']
    vaild_X = f_data['vaild_X']
    valid_Y = f_data['valid_Y']
    test_X = f_data['test_X']
    test_Y = f_data['test_Y']

    with open(cache_file_pickle, 'rb') as data_f_pickle:
        word2index, label2index = pickle.load(
            data_f_pickle)
    print("cache file load successful...")
    return word2index, label2index, train_X, train_Y, vaild_X, valid_Y, test_X, test_Y


if __name__ == "__main__":
    tf.app.run()
