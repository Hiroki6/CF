# -*- coding:utf-8 -*-

import sys
sys.dont_write_bytecode = True 
import numpy as np
import create_matrix
from fm import fm
from cythonfm import cylibfm

if __name__ == "__main__":
    print "データ作成"
    rate_matrix, labels, targets = create_matrix.create_matrix_dicVec()
    print "テストデータ作成"
    test_datas, targets, target_labels = create_matrix.divide_matrix(rate_matrix, labels, targets)
    print target_labels
    print test_datas
    print "FMクラス初期化"
    FM_obj = cylibfm.FM(target_data, labels, targets)
    print "学習開始"
    FM_obj.learning()
