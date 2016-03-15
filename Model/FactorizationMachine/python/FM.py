# -*- coding:utf-8 -*-
"""
Factorization Machineの実装
userID, itemID, 評価値, timestampを特徴量として学習する
学習の手法は交互最小二乗法(ALS)を用いる
アルゴリズムは
「Fast Context-aware Recommendations with Factorization Mahines」の論文のP.6を基に実装
"""
import numpy as np
import math

# 学習タイプ(ALSとSGD)
LEARNING_TYPE = {"ALS": 0, "SGD": 1}
class FM:
    def __init__(self, R, targets):
        self.R = R #評価値行列
        self.targets = targets # 教師配列
        self.n = len(self.R[0])
        self.N = len(self.R)

    def get_all_error(self):

        for data_index in xrange(self.N):
            self.get_error(data_index, self.targets[data_index])
            self.get_q_error(data_index)
    
    def get_q_error(self, data_index):

        for f in xrange(len(self.V[0])):
            for i in xrange(self.n):
                self.Q[data_index] += self.V[i][f] * self.R[data_index][i]

    def get_error(self, data_index, target):
        
        # 各特徴量の重みの和
        features = 0.0
        # 相互作用の重みの和
        iterations = 0.0
        for i in xrange(self.n):
            features += self.W[i] * self.R[data_index][i]
            for j in xrange(i+1, self.n):
                # 相互作用の特徴量の重みの和
                interaction_parameters = 0.0
                for f in xrange(len(self.V[0])):
                    interaction_parameters += self.V[i][f] * self.V[j][f]
                iterations += interaction_parameters * self.R[data_index][i] * self.R[data_index][j]

        self.E[data_index] = (self.w_0 + features + iterations) - target

    """
    w_0の更新
    """
    def update_global_bias(self):
        
        error_sum = 0.0
        for data_index in xrange(self.N):
            error_sum += self.E[data_index] - self.w_0
    
        new_w0 = -(error_sum) / (self.N + self.beta * error_sum)
        
        # 誤差の更新
        for data_index in xrange(self.N):
            self.E[data_index] += new_w0 - self.w_0

        self.w_0 = new_w0

    """
    Wの更新
    """
    def update_weight(self):
        
        for l in xrange(self.n):
            error_sum = 0.0
            feature_square_sum = 0.0
            for data_index in xrange(self.N):
                error_sum += (self.E[data_index] - self.W[l] * self.R[data_index][l]) * self.R[data_index][l]
                feature_square_sum += pow(self.R[data_index][l],2)
            new_wl = -(error_sum) / (feature_square_sum + self.beta*error_sum)
            for data_index in xrange(self.N):
                self.E[data_index] += (new_wl-self.W[l]) * self.R[data_index][l]
            self.W[l] = new_wl

    """
    Vの更新
    """
    def update_interaction(self):

        for f in xrange(len(self.V[0])):
            for l in xrange(self.n):
                error_sum = 0.0
                h_square_sum = 0.0
                for data_index in xrange(len(self.N)):
                    h_v = (self.R[data_index][l] * self.Q[data_index]) - (pow(self.R[data_index][l],2) * self.V[l][f])
                    error_sum += (self.E[data_index] - self.V[l][f] * h_v) * h_v
                    h_square_sum += pow(h_v,2)
                new_v = -(error_sum) / (h_square_sum + self.beta * error_sum)
                for data_index in xrange(len(self.N)):
                    self.E[data_index] += (new_v - self.V[l][f]) * self.R[data_index][l]
                    self.Q[data_index] += (new_v - self.V[j][f]) * self.R[data_index][l]
                self.V[l][f] = new_v
   

    def repeat_optimization():

        update_global_bias()
        update_weight()
        update_interaction()

    """
    ALSの学習
    """
    def learning(self, K = 5, beta = 0.2, step = 30):
        self.beta = beta
        # バイアス
        self.w_0 = 0
        # 重み
        self.W = np.zeros(self.n)
        # 相互作用の重み
        self.V = np.random.rand(self.n, K)
        # すべての誤差訓練データの誤差
        self.E = np.zeros(self.N)
        # すべてのVの重み誤差
        self.Q = np.zeros(self.N)

        """
        誤差の計算
        """
        self.get_all_error()
        for i in xrange(step):
            repeat_optimization()
