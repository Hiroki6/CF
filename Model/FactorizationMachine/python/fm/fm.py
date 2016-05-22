# -*- coding:utf-8 -*-
"""
Factorization Machineのpythonによる実装
userID, itemID, 評価値, timestampを特徴量として学習する
学習の手法は交互最小二乗法(ALS)を用いる
アルゴリズムは
「Fast Context-aware Recommendations with Factorization Mahines」の論文のP.6を基に実装
"""
import numpy as np
import math

LEARNING_TYPES = {"SGD": 0, "ALS": 1, "MCMC": 2}

class Fm:
    def __init__(self, R, labels, targets, seed=20, init_stdev=0.1):
        self.R = R #評価値行列
        self.labels = labels
        self.targets = targets # 教師配列
        self.n = len(self.R[0])
        self.N = len(self.R)
        self.seed = seed
        self.init_stdev = init_stdev

class FmAls(Fm):

    def __init__(self, R, labels, targets, seed=20, init_stdev=0.1):
        super(FmAls, self).__init__(R, labels, targets, seed, init_stdev)

    def _get_all_error(self):

        for data_index in xrange(self.N):
            print data_index
            self._get_error(data_index, self.targets[data_index])
            self._get_q_error(data_index)
    
    def _get_q_error(self, data_index):

        for f in xrange(len(self.V[0])):
            self.Q[data_index][f] = np.dot(self.V.T[f], self.R[data_index])

    def _get_error(self, data_index, target):
        
        # 各特徴量の重みの和
        features = 0.0
        # 相互作用の重みの和
        iterations = 0.0
        features = np.dot(self.W.T, self.R[data_index])
        for f in xrange(len(self.V[0])):
            iterations += pow(np.dot(self.V.T[f], self.R[data_index]),2) - np.dot(self.V.T[f]**2, self.R[data_index]**2)
        
        self.E[data_index] = (self.w_0 + features + iterations) - target
    """
    w_0の更新
    """
    def _update_global_bias(self):
        
        error_sum = np.sum(self.E) - self.w_0*self.N
        new_w0 = -(error_sum) / (self.N + self.beta * error_sum)
        
        # 誤差の更新
        vf = np.vectorize(self.add_value)
        self.E += new_w0 - self.w_0
        # for data_index in xrange(self.N):
        #     self.E[data_index] += new_w0 - self.w_0

        self.w_0 = new_w0
    
    """
    Wの更新
    """
    def _update_weight(self):
        
        for l in xrange(self.n):
            feature_square_sum = np.sum(self.R[:,l] ** 2)
            error_sum = np.sum((self.E - self.W[l] * np.transpose(self.R)[l]) * np.transpose(self.R)[l])
            feature_square_sum = np.sum(self.R.T[l]**2)
            new_wl = -(error_sum) / (feature_square_sum + self.beta*error_sum)
            self.E += (new_wl - self.W[l]) * np.transpose(self.R)[l]
            self.W[l] = new_wl

    """
    Vの更新
    """
    def _update_interaction(self):

        for f in xrange(len(self.V[0])):
            for l in xrange(self.n):
                error_sum = 0.0
                h_square_sum = 0.0
                for data_index in xrange(self.N):
                    h_v = (self.R[data_index][l] * self.Q[data_index]) - (pow(self.R[data_index][l],2) * self.V[l][f])
                    error_sum += (self.E[data_index] - self.V[l][f] * h_v) * h_v
                    h_square_sum += pow(h_v,2)
                new_v = -(error_sum) / (h_square_sum + self.beta * error_sum)
                self.E += (new_v - self.V[l][f]) * np.transpose(self.R)[l]
                np.transpose(self.Q)[f] += (new_v - self.V[l][f]) * np.transpose(self.R)[l]
                self.V[l][f] = new_v
   

    def _repeat_optimization():

        self._update_global_bias()
        self._update_weight()
        self._update_interaction()
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
        np.random.seed(seed=self.seed)
        self.V = np.random.normal(scale=self.init_stdev,size=(self.n, K))
        # すべての誤差訓練データの誤差
        self.E = np.zeros(self.N)
        # すべてのVの重み誤差
        self.Q = np.zeros((self.N, K))
        
        """
        誤差の計算
        """
        self._get_all_error()
        for i in xrange(step):
            print i
            self._repeat_optimization()


class FmSgd(Fm):
    def __init__(self, R, labels, targets, seed=20, init_stdev=0.1):
        super(FmAls, self).__init__(R, labels, targets, seed, init_stdev)

