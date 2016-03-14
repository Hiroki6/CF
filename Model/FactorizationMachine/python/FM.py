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

    def get_error(self, x_index, target):
        
        # 各特徴量の重みの和
        features = 0.0
        # 相互作用の重みの和
        iterations = 0.0
        for i in xrange(len(self.R[0])):
            features += self.W[i] * self.R[x_index][i]
            for j in xrange(i+1, len(self.R[0])):
                # 相互作用の特徴量の重みの和
                interaction_parameters = 0.0
                for f in xrange(K):
                    interaction_parameters += self.V[i][f] * self.V[j][f]
                iterations += interaction_parameters * self.R[x_index][i] * self.R[x_index][j]

        self.error = (self.w_0 + features + iterations) - target

    def learning(self, K = 5):
        # バイアス
        self.w_0 = 0
        # 重み
        self.W = np.zeros(len(self.R[0]))
        # 相互作用の重み
        self.V = np.random.rand(len(self.R[0]),K)
        # 誤差
        self.error = 0
