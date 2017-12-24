# -*- coding:utf-8 -*-

import math
import random
import numpy as np
import sys
sys.dont_write_bytecode = True 

"""
@return(element_dic) {"値":id}の辞書
"""
def create_element_map(filepass):

    element_dic = {}
    id = 0
    with open(filepass) as f:
        for line in f:
            element = line.replace("\n","").split('|')[0]
            element_dic[element] = id
            id += 1

    return element_dic


def create_ratelist(filepass):

    ret = []
    
    with open(filepass) as f:
        for line in f:
            ret.append(line.replace("\n","").split('\t'))

    return ret

"""
テストデータを300個作る
@param(RateArray) キーにユーザーとアイテムを持つ二次元ディクショナリ
@return(RateArray) テストデータを除いた教師データ
@return(testData) キーにユーザーとアイテムを持つ300個のテスト用データディクショナリ
"""
def create_test_data(rate_matrix):
  
  testData = []
  count = 0
  while count < 300:
      user = random.randint(0, len(rate_matrix)-1)
      item = random.randint(0, len(rate_matrix[0])-1)
      rate = rate_matrix[user][item]
      if rate < 1.0:
          continue
      test = [user, item, rate]
      testData.append(test)
      rate_matrix[user][item] = 0
      count += 1

  return rate_matrix, testData

def create_test_data_by_testfile(usermap, itemmap):
    
    ratelist = create_ratelist("../../data/ml-100k/u1.test")
    test_matrix = np.zeros((len(usermap),len(itemmap)))
    test_data = []
    for rate in ratelist:
        user_id = int(usermap[rate[0]])
        item_id = int(itemmap[rate[1]])
        test = [user_id, item_id, int(rate[2])]
        test_data.append(test)

    return test_data

"""
numpy.array化する
"""
def create_matrix(train_file):

    usermap = create_element_map("../../data/ml-100k/u.user")
    itemmap = create_element_map("../../data/ml-100k/u.item")
    # userID::movieID::rating::timestamp
    ratelist = create_ratelist(train_file)
   
    rate_matrix = np.zeros((len(usermap),len(itemmap)))
    for rate in ratelist:
        user_id = usermap[rate[0]]
        item_id = itemmap[rate[1]]
        rate_matrix[user_id][item_id] = rate[2]

    return rate_matrix, usermap, itemmap
