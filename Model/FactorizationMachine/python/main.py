# -*- coding:utf-8 -*-

import sys
sys.path.append("cython-fm")
sys.path.append("fm")
import numpy as np
import create_matrix
import fm
import cylibfm

"""
学習データとテストデータに分ける
"""
def create_test_data(target_data):


    return

if __name__ == "__main__":
    print "データ作成"
    target_data, labels, targets = create_matrix.create_matrix_dicVec()
    print labels
    print "FMクラス初期化"
    FM_obj = cylibfm.FM(target_data, labels, targets)
    print "学習開始"
    FM_obj.learning()
