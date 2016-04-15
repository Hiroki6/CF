# -*- coding:utf-8 -*-

import numpy as np
from sklearn.feature_extraction import DictVectorizer
import random

"""
return(rate_matrix) FM用のデータ
"""
def create_matrix_dicVec():

    ratelist = create_ratelist("../../../data/ml-1m/ratings.dat")
    
    rate_array= []
    targets = [] # 教師データ
    for rating in ratelist:
        rate_dic = {}
        user = rating[0]
        movie = rating[1]
        rate_dic["user"] = user
        rate_dic["movie"] = movie
        rate_dic["time"] = (int(rating[3])%900000000)/1000000
        targets.append(int(rating[2]))
        rate_array.append(rate_dic)
   
    v = DictVectorizer()
    X = v.fit_transform(rate_array)
    rate_matrix = X.toarray()
    labels = v.get_feature_names()
    targets = np.array(targets)

    return rate_matrix, labels, targets

def create_element(filepass):

    ret = []
    with open(filepass) as f:
        for line in f:
            ret.append(line.replace("\n","").split('::')[0])
    
    return ret


def create_ratelist(filepass):

    ret = []

    with open(filepass) as f:
        for line in f:
            ret.append(line.replace("\n","").split('::'))

    return ret

"""
userを8:2に分割する
@return(test_users) テスト用のユーザー
"""
def create_test_user():

    userlist = create_element("../../../data/ml-1m/users.dat")
    test_users = []
    number_of_test = int(len(userlist) * 0.2)
    for i in xrange(number_of_test):
        index = random.randint(0, len(userlist)-1)
        user = userlist.pop(index)
        test_users.append(user)

    return test_users


"""
rate_matrixを学習データとテストデータに分ける
ユーザー配列からユーザーをランダムに20%取り出す
その中からそのユーザーの半分のratingをテストデータとして使う
@returns(test_matrix) テスト用のデータ{user_index: {rate_index: []}}
@returns(targets) テスト用のラベルを0.0にした学習用ラベル
@returns(test_labels) 実際のラベルとtargetのインデックスを保存したラベル{rate_index: label}
"""
def divide_matrix(rate_matrix, labels, targets):

    test_users = create_test_user()
    test_matrix = {}
    test_labels = {}
    for test_user in test_users:
        count = 0
        user_index = labels.index("user="+test_user)
        test_matrix.setdefault(user_index, {})
        for rate_index in xrange(len(rate_matrix)):
            if count >= 5:
                break
            if rate_matrix[rate_index][user_index] == 1.0:
                test_matrix[user_index][rate_index] = rate_matrix[rate_index]
                test_labels[rate_index] = targets[rate_index]
                targets[rate_index] = 0.0
                count += 1

    return test_matrix, targets, test_labels
