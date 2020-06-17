
import math
import heapq

_sess = None
_model = None
_testRatings = None
_testNegatives = None
_K = None
_dish_to_category = None


def evaluate_model(sess, model, testRatings, testNegatives, K, dish_to_category):
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
    negative_instance = _testNegatives[str(user)][50:100]
    for negative in negative_instance:
        predict_items.append(negative)
        user_input_index.append(user)
        item_input_index.append(negative)
        categories.append(_dish_to_category[str(negative)])
        labels.append(0)

    map_item_score = {}

    feed_dict = {_model.user_input: user_input_index, _model.item_input: item_input_index,
                 _model.labels: labels, _model.categories: categories, _model.dropout_keep_prob: 1.0,
                 _model.is_training_flag: False}
    predictions = _sess.run([_model.logits], feed_dict)
    predictions = predictions[0]
    for i in range(len(predict_items)):
        map_item_score[predict_items[i]] = predictions[i]
    
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, positive_instance)
    ndcg = getNDCG(ranklist, positive_instance)
    return hr, ndcg


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
