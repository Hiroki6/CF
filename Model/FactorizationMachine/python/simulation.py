# -*- coding:utf-8 -*-
"""
Factorization Machineのシミューレションを実行するファイル
"""
import sys
sys.dont_write_bytecode = True 
import numpy as np
import create_matrix
import math

"""
@params(user) テストするユーザー
@params(test_items) アイテム
@params(data_labels) 学習で用いられたラベルデータ
@params(test_matrix) テスト用の行列
"""
def create_test_matrix(user, test_items, data_labels):

    user_index = data_labels.index("user="+user)
    test_matrix = np.zeros((len(test_items), len(data_labels)))
    for index, item in enumerate(test_items):
        item_index = data_labels.index("item="+item)
        test_matrix[index][user_index] = 1.0
        test_matrix[index][item_index] = 1.0

    return test_matrix

"""
そのユーザーについて学習されていないアイテム集合
"""
def create_items_except_learning_by_user(user, test_items):

    ratelist = create_matrix.create_ratelist("../../../data/ml-1m/ratings.dat")
    itemlist = create_matrix.create_element("../../../data/ml-1m/movies.dat")
    for index, item in enumerate(itemlist):
        if ratelist[user].has_key(item) or test_items.has_key(item):
            itemlist.pop(index)

    return itemlist

def simulation(learn_obj, test_data, data_labels):

    map_k_num = 0.0
    for user, values in test_data.items():
        test_items = create_items_except_learning_by_user(user, test_data[user])
        test_matrix = create_test_matrix(user, test_items, data_labels)
        rankings = learn_obj.recommendations(test_matrix, test_items)
        map_k_num += evaluation_rankings(rankings, test_data[user])

    return map_k_num * 100 / len(test_data)

def evaluation_rankings(rankings, test_items):
    
    ap_num = 0.0
    for rank, content in enumerate(rankings):
        item = content[1]
        if item in test_items:
            print content
            ap_num += pow(2, (-rank+1)/5)
    ap = ap_num / 5
    return ap

def calc_rmse(fm_obj, test_matrixs, test_targets):

    print "精度計測開始"
    sum_error = 0.0
    for index, test_matrix in enumerate(test_matrixs):
        sum_error += pow((fm_obj.cython_FM.predict(test_matrix) - test_targets[index]), 2)
        print index

    rmse = math.sqrt(sum_error/len(test_matrix))
    return rmse

