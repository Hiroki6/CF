# -*- coding:utf-8 -*-

import sys
sys.dont_write_bytecode = True 
import numpy as np
import create_matrix
import simulation
from fm import fm
from cythonfm import cylibfm

def main():
    print "データ作成"
    learn_matrix, test_data, labels, targets = create_matrix.create_matrix_dicVec()
    for user, values in test_data.items():
        itemlist = simulation.create_items_except_learning_by_user(user, test_data[user])
        print len(itemlist)
    print "FMクラス初期化"
    FM_obj = cylibfm.FM(learn_matrix, labels, targets)
    print "学習開始"
    FM_obj.learning()

if __name__ == "__main__":
    main()
