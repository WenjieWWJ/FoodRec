'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
import random
import tensorflow as tf
from time import time
#from numba import jit, autojit

# Global variables that are shared across processes
# 全局变量是在整个python文件中声明，全局范围内都可以访问
_sess = None
_model = None
_testRatings = None
_testNegatives = None
_K = None
_dish_to_category = None


def evaluate_model(sess, model, testRatings, testNegatives, K, dish_to_category):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _sess
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _dish_to_category
    _sess = sess
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    _dish_to_category = dish_to_category
        
    hits, ndcgs = [], []
    for idx in _testRatings:
        (hr, ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
    # print('hits:', hits)
    # print('ndcgs:', ndcgs)
    # for idx in range(len(_testRatings)):
    #     (hr,ndcg) = eval_one_rating(idx)
    #     hits.append(hr)
    #     ndcgs.append(ndcg)
    return hits, ndcgs


def eval_one_rating(user):
    user_input_index, item_input_index, labels, categories, predict_items = [], [], [], [], []
    if not str(user) in _testRatings or len(_testRatings[str(user)]) == 0:
        return
    positive_instance = _testRatings[str(user)][0]
    predict_items.append(positive_instance)
    user_input_index.append(user)
    item_input_index.append(positive_instance)
    categories.append(_dish_to_category[str(positive_instance)])
    labels.append(1)
    negative_instance = _testNegatives[str(user)][50:]
    for negative in negative_instance:
        predict_items.append(negative)
        user_input_index.append(user)
        item_input_index.append(negative)
        categories.append(_dish_to_category[str(negative)])
        labels.append(0)

    # Get prediction scores
    map_item_score = {}

    feed_dict = {_model.user_input: user_input_index, _model.item_input: item_input_index,
                 _model.labels: labels, _model.categories: categories, _model.dropout_keep_prob: 1.0,
                 _model.is_training_flag: False}
    predictions = _sess.run([_model.logits], feed_dict)
    predictions = predictions[0]
    # print('predictions:', predictions)
    # print('predictions[0]:', predictions)
    # print('predict_items:', predict_items)
    for i in range(len(predict_items)):
        map_item_score[predict_items[i]] = predictions[i]
    # print('map_item_score:', map_item_score)
    # items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    # print('ranklist:', ranklist)
    # print('gtItem:', gtItem)
    hr = getHitRatio(ranklist, positive_instance)
    ndcg = getNDCG(ranklist, positive_instance)
    return hr, ndcg

def getHitRatio(ranklist, gtItem):
    # gtItem = gtItem[0]
    # 只要用于测试的正例出现在了predictions所有分数前K（前10）的位置，HR的值就是1，代表hit了
    for item in ranklist:
        # print('item, gtItem:', item, gtItem)
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    # gtItem = gtItem[0]
    # 返回的NDCG指标值的大小还与test的正例位于predictions中的位置有关，越靠前，返回的NDCG值越大，说明预测和推荐的越准确，越靠后，
    # 说明越不准确，返回的NDCG值越小
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
