# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

def create_matrix():

    userlist = create_element("../../../data/ml-1m/users.dat")
    itemlist = create_element("../../../data/ml-1m/movies.dat")
    # userID::movieID::rating::timestamp
    ratelist = create_ratelist("../../../data/ml-1m/ratings.dat")
   
    watch_itemlist = add_column_name(itemlist, "I")
    time_list = ["time"]
    rate_matrix = pd.DataFrame(np.zeros((len(ratelist), len(userlist)+len(watch_itemlist)+1)), columns = userlist+watch_itemlist+time_list)
    for rate_index in range(len(rate_matrix)):
        # 0行目の処理
        if rate_index == 0:
            rating = rate_matrix[:1]
        # 0行目以外の処理
        else:
            rating = rate_matrix[rate_index:rate_index+1]

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
    #print rate_matrix
