# -*- coding:utf-8 -*-
"""
Factorization Machineのcythonによる実装
userID, itemID, 評価値, timestampを特徴量として学習する
学習の手法は交互最小二乗法(ALS)を用いる
アルゴリズムは
「Fast Context-aware Recommendations with Factorization Mahines」の論文のP.6を基に実装
"""
import numpy as np
import math
import cy_fm_als

class CyFmAls:
    def __init__(self, R, labels, targets):
        self.R = R #評価値行列
        self.labels = labels
        self.targets = targets # 教師配列
        self.n = len(self.R[0])
        self.N = len(self.R)

    def learning(self, K = 5, beta = 0.2, step = 30):
        # バイアス
        self.w_0 = 0
        # 重み
        self.W = np.zeros(self.n)
        # 相互作用の重み
        self.V = np.random.rand(self.n, K)
        # すべての誤差訓練データの誤差
        self.E = np.zeros(self.N)
        # すべてのVの重み誤差
        self.Q = np.zeros((self.N, K))
        self.cython_FM = cy_fm_als.CyFmAls(self.R, self.targets, self.W, self.V, self.E, self.Q, self.w_0, beta, self.n, self.N, K, step)
        self.cython_FM.learning()

    def recommendations(self, test_matrix, items, rank = 10):
        
        rankings = [(self.cython_FM.predict(test), item) for test, item in zip(test_matrix, items)]
        rankings.sort()
        rankings.reverse()
        return rankings[::rank]
