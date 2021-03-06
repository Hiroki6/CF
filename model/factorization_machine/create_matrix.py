# -*- coding:utf-8 -*-

import numpy as np
from sklearn.feature_extraction import DictVectorizer
import random

"""
シミュレーション用のデータ作成
@returns(learn_matrix) 学習用データ
@returns(test_data) テスト用データ{user: [items]}
@returns(labels) 学習データの列ラベル
@returns(targets) 学習ラベル
"""
def create_matrix_dicVec():

    ratelist = create_ratelist("../../data/ml-100k/u1.base")
    test_ratelist = create_ratelist("../../data/ml-100k/u1.test")
    
    rate_array= []
    targets = [] # 教師データ
    regs_data = {}
    print "正規化項用データ作成"
    test_nums = 0
    for user, values in ratelist.items():
        regs_data.setdefault(user, {})
        index = random.randint(0, len(ratelist[user])-1)
        item = random.choice(ratelist[user].keys())
        rate = ratelist[user].pop(item)
        regs_data[user][item] = rate
        test_nums += 1

    # 学習用データ作成
    print "学習用データ作成"
    for user, values in ratelist.items():
        for item, rate in values.items():
            rate_dic = {}
            rate_dic["user"] = user
            rate_dic["item"] = item
            targets.append(rate)
            rate_array.append(rate_dic)
    
    # FM用に変形
    v = DictVectorizer()
    X = v.fit_transform(rate_array)
    learn_matrix = X.toarray()
    labels = v.get_feature_names()
    targets = np.array(targets)

    print "テストデータ作成"
    test_matrix, test_targets = create_test_matrix(test_ratelist, labels, 20000)
    regs_matrix, regs_targets = create_test_matrix(regs_data, labels, test_nums)

    return learn_matrix, targets, test_matrix, test_targets, regs_matrix, regs_targets

def create_element(filepass):

    ret = []
    with open(filepass) as f:
        for line in f:
            ret.append(line.replace("\n","").split('\t')[0])
    
    return ret

def create_ratelist(filepass):

    rate_dic = {}

    with open(filepass) as f:
        for line in f:
            rating = line.replace("\n","").split('\t')
            user = rating[0]
            item = rating[1]
            rate = int(rating[2])
            if not rate_dic.has_key(user):
                rate_dic.setdefault(user, {})
            rate_dic[user][item] = rate

    return rate_dic


"""
userを8:2に分割する
@return(test_users) テスト用のユーザー
"""
def create_test_user():

    userlist = create_element("../../data/ml-100k/u.user")
    test_users = []
    number_of_test = int(len(userlist) * 0.2)
    for i in xrange(number_of_test):
        index = random.randint(0, len(userlist)-1)
        user = userlist.pop(index)
        test_users.append(user)

    return test_users

"""
テストデータのFM配列作成
"""
def create_test_matrix(test_data, data_labels, test_nums):

    test_matrix = np.zeros((test_nums, len(data_labels)))
    col = 0
    test_targets = []
    for user, values in test_data.items():
        user_index = data_labels.index("user="+user)
        not_learn_items = []
        for item, rate in values.items():
            item_label_name = "item=" + item
            if item_label_name not in data_labels:
                not_learn_items.append(item)
            else:
                item_index = data_labels.index(item_label_name)
                test_matrix[col][user_index] = 1.0
                test_matrix[col][item_index] = 1.0
                test_targets.append(rate)
                col += 1

        for not_learn_item in not_learn_items:
            test_data[user].pop(not_learn_item)
    test_matrix = test_matrix[:col]
    test_targets = np.array(test_targets)

    return test_matrix, test_targets

