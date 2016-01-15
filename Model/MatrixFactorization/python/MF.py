# -*- coding:utf-8 -*-
"""
SVD++の実装
参考文献:
    「Advances in Collaborative Filtering」,Y.Koran and R.Bell ,2007 IEEE
R: ユーザーとアイテムをインデックスにもつ二次元ディクショナリ
"""
import numpy as np

class basicMF:
    def __init__(self, R):
        self.R = R # 評価値行列
    
    def get_rating_error(self, r, p, q):
        return r- np.dot(p, q)

    def get_error(self, beta):
        """
        目的関数
        @param(beta) 正規化係数
        """
        self.error = 0.0

        for user_index in self.R:
            for item_index in self.R[user_index]:
                if self.R[user_index][item_index] == 0:
                    continue
                self.error += pow(self.get_rating_error(self.R[user_index][item_index], self.P[:,user_index], self.Q[:,item_index]), 2)
        # 正規化項
        self.error += beta * (np.linalg.norm(self.P) + np.linalg.norm(self.Q))
    
    def learning(self, K, steps = 5000, gamma = 0.005, beta = 0.02, threshold = 0.1):
        """
        ユーザー行列Pとアイテム行列Qを最適化によって求める
        @param(K) 疑似行列の次元
        @param(steps) 学習回数
        @param(alpha) 学習係数
        @param(beta) 正規化係数
        @param(threshold) 誤差の閾値
        """
        self.P = np.random.rand(K, len(self.R))
        self.Q = np.random.rand(K, len(self.R[0]))

        for step in xrange(steps):
            for user_index in self.R:
                for item_index in self.R[user_index]:
                    if self.R[user_index][item_index] == 0:
                        continue
                    err = self.get_rating_error(self.R[user_index][item_index], self.P[:,user_index], self.Q[:,item_index])
                    for k in xrange(K):
                        self.P[k][user_index] += gamma * (err*self.Q[k][item_index] - beta*self.P[k][user_index])
                        self.Q[k][item_index] += gamma * (err*self.P[k][user_index] - beta*self.Q[k][item_index])
            self.get_error(beta)
            print self.error
            if self.error < threshold:
                self.nR = np.dot(self.P.T, self.Q) # 得られた評価値行列
                break

    def predict(self, user, item):
        return self.nR[int(user)-1][int(item)-1]

    def recommends(self, user):
        user_index = int(user) - 1
        rankings = [(self.nR[user_index][item_index], str(item_index+1)) for item_index in self.R[user_index] if self.R[user_index][item_index] == 0]
        rankings.sort()
        return rankings


class svd(basicMF):

    def __init__(self, R):
        self.R = R # 評価値行列
        self.myu = get_ave()

    def get_ave(self):

        count = 0
        sumRate = 0
        for user_index in self.R:
            for item_index in self.R[user_index]:
                if not self.R[user_index][item_index] == 0:
                    count += 1
                    sumRate += self.R[user_index][item_index]

        if count == 0:
            return 0
        else:
            return sumRate/count
        
    def get_rating_error(self, r, p, q, b_u, b_i):
        return r- (self.myu + b_i + b_u + np.dot(p, q))

    def get_error(self, beta):
        """
        目的関数
        @param(beta) 正規化係数
        """
        self.error = 0.0

        for user_index in self.R:
            for item_index in self.R[user_index]:
                if R[user_index][item_index] == 0:
                    continue
                self.error += pow(self.get_rating_error(self.R[user_index][item_index], self.P[:,user_index], self.Q[:,item_index], self.B_u[user_index], self.B_i[item_index]), 2)
        # 正規化項
        self.error += beta * (np.linalg.norm(self.B_u) + np.linalg.norm(self.B_i) + np.linalg.norm(self.P) + np.linalg.norm(self.Q))
    
    def learning(self, K, steps = 5000, gamma = 0.005, beta = 0.02, threshold = 0.01):
        """
        ユーザー行列Pとアイテム行列Qを最適化によって求める
        @param(K) 疑似行列の次元
        @param(steps) 学習回数
        @param(alpha) 学習係数
        @param(beta) 正規化係数
        @param(threshold) 誤差の閾値
        """
        self.P = np.random.rand(K, len(self.R)) # ユーザー嗜好ベクトル
        self.Q = np.random.rand(K, len(self.R[0])) # アイテム嗜好ベクトル
        self.B_u = np.random.rand(len(self.R)) # ユーザーバイアス
        self.B_i = np.random.rand(len(self.R[0])) # アイテムバイアス

        for step in xrange(steps):
            for user_index in self.R:
                for item_index in self.R[user_index]:
                    if self.R[user_index][item_index] == 0:
                        continue
                    err = self.get_rating_error(self.R[user_index][item_index], self.P[:,user_index], self.Q[:,item_index], self.B_u[user_index], self.B_i[item_index])
                    for k in xrange(K):
                        self.P[k][user_index] += gamma * (err*self.Q[k][item_index] - beta*self.P[k][user_index])
                        self.Q[k][item_index] += gamma * (err*self.P[k][user_index] - beta*self.Q[k][item_index])
                        self.B_u[user_index] += gamma * (err - beta*self.B_u[user_index])
                        self.B_i[item_index] += gamma * (err - beta+self.B_i[item_index])
            self.get_error(beta)
            print self.error
            if self.error < threshold:
                self.nR = np.dot(self.P.T, self.Q) # 得られた評価値行列
                break
