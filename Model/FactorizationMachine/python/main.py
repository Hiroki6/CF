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
    return create_matrix.create_matrix_dicVec()

def learn_fm_als(target_data, labels, targets):
    print "FMクラス初期化"
    fm_obj = alslibfm.CyFmAls(target_data, labels, targets)
    print "ALSで学習開始"
    fm_obj.learning(step=1)
    return fm_obj

def learn_fm_sgd(learn_data, regs_matrix, targets, regs_targets):
    print "FMクラス初期化"
    fm_obj = sgdlibfm.CyFmSgd(learn_data, regs_matrix, targets, regs_targets)
    print "SGDで学習開始"
    fm_obj.learning(0.005, step=10)
    return fm_obj

if __name__ == "__main__":
    learn_data, targets, test_matrix, test_targets, regs_matrix, regs_targets = setup()
    fm_obj = learn_fm_sgd(learn_data, regs_matrix, targets, regs_targets)
    """
    更に学習する場合
    fm_obj.cython_FM.repeat_optimization()
    """
    # rmseの評価
    rmse = simulation.calc_rmse(fm_obj, test_matrix, test_targets)
    print rmse
