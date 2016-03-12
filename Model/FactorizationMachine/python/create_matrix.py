# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

def create_matrix():

    userlist = create_element("../../../data/ml-1m/users.dat")
    itemlist = create_element("../../../data/ml-1m/movies.dat")
    # userID::movieID::rating::timestamp
    ratelist = create_ratelist("../../../data/ml-1m/ratings.dat")
   
    watch_itemlist = add_column_name(itemlist, "I")
    time_list = ["time"]
    target_list = ["target"]
    rate_matrix = pd.DataFrame(np.zeros((len(ratelist), len(userlist)+len(watch_itemlist)+2)), columns = userlist+watch_itemlist+time_list+target_list)
    for rate_index in xrange(len(rate_matrix)):
        input_rating = ratelist[rate_index]
        rate_matrix[input_rating[0]][rate_index] = 1
        rate_matrix["I" + input_rating[1]][rate_index] = 1
        print rate_index

    return rate_matrix
"""
return(rate_matrix) FM用のデータ
"""
def create_matrix_dicVec():

    ratelist = create_ratelist("../../../data/ml-1m/ratings.dat")
    
    rate_array= []
    for rating in ratelist:
        rate_dic = {}
        rate_dic["user"] = rating[0]
        rate_dic["item"] = rating[1]
        rate_dic["time"] = (int(rating[3])%900000000)/1000000
        rate_dic["rating"] = int(rating[2])
        rate_array.append(rate_dic)
    
    v = DictVectorizer()
    X = v.fit_transform(rate_array)
    rate_matrix = X.toarray()
    labels = v.get_feature_names()

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
    rate_matrix = create_matrix_dicVec()

