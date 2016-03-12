# -*- coding:utf-8 -*-
"""
Factorization Machineの実装
userID, itemID, 評価値, timestampを特徴量として学習する
"""
import numpy as np

# 学習タイプ(ALSとSGD)
LEARNING_TYPE = {"ALS": 0, "SGD": 1}
class FM:
    def __init__(self, R):
        self.R = R #評価値行列

    def get_error(self):
        
        return

    def learning(self, K = 5):
        # バイアス
        self.w_0 = 0
        # 重み
        self.W = np.random.rand(len(self.R[0]))
        # 相互作用の重み
        self.V = np.random.rand(len(self.R[0]),K)
