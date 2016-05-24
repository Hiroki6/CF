# -*- coding:utf-8 -*-

import sys
sys.dont_write_bytecode = True 
import numpy as np
import create_matrix
import simulation
from CyFmAls import alslibfm
from CyFmSgd import sgdlibfm

def setup():

    print "データ作成"
    target_data, test_matrix, test_targets, test_data, labels, targets = create_matrix.create_matrix_dicVec()
    return target_data, test_matrix, test_targets, test_data, labels, targets

def learn_fm_als(target_data, labels, targets):
    print "FMクラス初期化"
    FM_obj = alslibfm.CyFmAls(target_data, labels, targets)
    print "ALSで学習開始"
    FM_obj.learning(step=1)
    return FM_obj

def learn_fm_sgd(target_data, test_matrix, labels, targets, test_targets):
    print "FMクラス初期化"
    FM_obj = sgdlibfm.CyFmSgd(target_data, test_matrix, labels, targets, test_targets)
    print "SGDで学習開始"
    FM_obj.learning(0.005, step=1)
    return FM_obj

if __name__ == "__main__":
    target_data, test_matrix, test_targets, test_data, labels, targets = setup()
    FM_obj = learn_fm_sgd(target_data, test_matrix, labels, targets, test_targets)
    print "精度評価"
    map_value = simulation.simulation(FM_obj, test_data, labels)
    """
    更に学習する場合
    FM_obj.cython_FM.repeat_optimization()
    """
    print map_value

