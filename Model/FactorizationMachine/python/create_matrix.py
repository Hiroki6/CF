# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

def create_matrix():

    userlist = create_element("../../../data/ml-1m/users.dat")
    itemlist = create_element("../../../data/ml-1m/movies.dat")
    # userID::movieID::rating::timestamp
    ratelist = create_ratelist("../../../data/ml-1m/ratings.dat")
   
    watch_itemlist = add_column_name(itemlist, "I")
    inverse_itemlist = add_column_name(itemlist, "IN")
    user_rate_matrix = pd.DataFrame(np.zeros((len(ratelist), len(userlist))), index=range(len(ratelist)), columns=userlist)
    item_rate_matrix = pd.DataFrame(np.zeros((len(ratelist), len(watch_itemlist))), index=range(len(ratelist)), columns=watch_itemlist)
    inverse_item_rate_matrix = pd.DataFrame(np.zeros((len(ratelist), len(inverse_itemlist))), index=range(len(ratelist)), columns=inverse_itemlist)
    #rate_matrix = pd.merge(user_rate_matrix, item_rate_matrix,right_index=True, left_index=True, how="outer") 
    #rate_matrix = pd.merge(rate_matrix, inverse_item_rate_matrix, right_index=True, left_index=True, how="outer")
    rate_matrix = pd.concat([user_rate_matrix, item_rate_matrix, inverse_item_rate_matrix], axis=1)
    
    # time_row = 1
    # item_count = len(itemlist)
    # user_count = len(userlist)
    # # ユーザーとratingの数のdict
    # user_rate_count = {}
    # for rate in ratelist:
    #     if not user_rate_count.has_key(int(rate[0])):
    #         user_rate_count.setdefault(int(rate[0]), 0)
    #     user_rate_count[rate[0]] += 1
    # print user_rate_count
    # # ユーザー, アイテム, ユーザーの評価したアイテム数の逆数, 時間, 前に付けたアイテム, 評価値
    # row = user_count + item_count + item_count + 1 + 1
    # # 縦軸X, 横軸各情報
    # rate_matrix = np.zeros((len(ratelist),row))
    # for rate, i in zip(ratelist, range(len(ratelist))):
    #     rate_matrix[i][int(rate[0])-1] = 1
    #     rate_matrix[i][6040+int(rate[1])-1] = 1
    #     rate_matrix[i][row-1] = int(rate[2])
    return rate_matrix

def create_element(filename):

    ret = []
    for line in open(filename):
        ret.append(line.replace("\n","").split('::')[0])
    
    return ret


def create_ratelist(filename):

    ret = []

    for line in open(filename):
        ret.append(line.replace("\n","").split('::'))

    return ret

"""
listの各要素の先頭にadd_nameを加える
@params(changed_list) list
@params(add_name) 加える名前
"""
def add_column_name(changed_list, add):

    for index in range(len(changed_list)):
        changed_list[index] = add+changed_list[index]

    return changed_list

if __name__ == "__main__":
    rate_matrix = create_matrix()
    print rate_matrix
