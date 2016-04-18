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
    print "FMクラス初期化"
    FM_obj = cylibfm.FM(learn_matrix, labels, targets)
    print "学習開始"
    FM_obj.learning()
    print "精度評価"
    map_value = simulation.simulation(FM_obj, test_data, labels, tag_map)
    print map_value


if __name__ == "__main__":
    main()
