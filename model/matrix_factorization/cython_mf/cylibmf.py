# -*- coding:utf-8 -*-
"""
Matrix Factorizationのcythonによる実装
参考文献:
    「Advances in Collaborative Filtering」,Y.Koran and R.Bell ,2007 IEEE
R: ユーザーとアイテムをインデックスにもつ二次元ディクショナリ
"""
import numpy as np
import cy_mf as cMF
import time
import sys
sys.dont_write_bytecode = True 

"""
cython化したMF
"""
class CythonMF:
    
    def __init__(self, R):
        self.R = R # 評価値行列
        self.u_num = len(R)
        self.i_num = len(R[0])

    def learning(self, K, steps = 100, gamma = 0.005, beta = 0.02, threshold = 0.1):

        self.P = np.random.rand(K, self.u_num)
        self.Q = np.random.rand(K, self.i_num)
        self.cython_obj = cMF.FastMF(self.R, self.P, self.Q, self.u_num, self.i_num, K, steps, gamma, beta, threshold)
        start_time = time.time()
        self.cython_obj.learning()
        end_time = time.time()
        print end_time - start_time

    def predict(self, user, item):
        return self.cython_obj.predict(user, item)



