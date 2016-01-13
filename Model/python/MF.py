# -*- coding:utf-8 -*-
"""
SVD++の実装
参考文献:
    「Advances in Collaborative Filtering」,Y.Koran and R.Bell ,2007 IEEE
"""
import numpy

class basicMF:
    def __init__(self, R):
        self.R = R # 評価値行列
    
    def get_rating_error(r, p, q):
        return r- numpy.dot(p, q)

    def get_error(self, P, Q, beta):
        """
        目的関数
        @param(beta) 正規化係数
        """
        self.error = 0.0

        for user in range(len(self.R)):
            for item in xrange(len(self.R[user])):
                if R[user][item] == 0:
                    continue
                self.error += pow(self.get_rating_error(self.R[user][item], self.P[:,user], self.Q[:,item]), 2)
        # 正規化項
        self.error += beta * (numpy.linalg.norm(self.P) + numpy.linalg.norm(self.Q))
    
    def learning(self, K, steps = 5000, gamma = 0.0002, beta = 0.02, threshold = 0.001):
        """
        ユーザー行列Pとアイテム行列Qを最適化によって求める
        @param(K) 疑似行列の次元
        @param(steps) 学習回数
        @param(alpha) 学習係数
        @param(beta) 正規化係数
        @param(threshold) 誤差の閾値
        """
        self.P = numpy.random.rand(K, len(self.R))
        self.Q = numpy.random.rand(K, len(self.R[0]))

        for step in xrange(steps):
            for user in xrange(len(self.R)):
                for item in xrange(len(self.R[user])):
                    if self.R[user][item] == 0:
                        continue
                    err = self.get_rating_error(self.R[user][item], self.P[:,user], self.Q[:,item])
                    for k in xrange(K):
                        self.P[k][user] += gamma * (err*self.Q[k][item] - beta*self.P[k][user])
                        self.Q[k][item] += gamma * (err*self.P[k][user] - beta*self.Q[k][item])
            self.get_error(beta)
            if self.error < threshold:
                self.nR = numpy.dot(self.P.T, self.Q) # 得られた評価値行列
                break

    def predict(self, user, item):
        return self.nR[user][item]

    def recommends(self, user):
        rankings = [(self.nR[user][item], item) for item in self.R[user] if self.R[user][item] == 0]
        rankings.sort()
        return rankings


class svd(basicMF):

    def __init__(self, R):
        self.R = R # 評価値行列
        self.myu = get_ave(self)

    def get_ave(self):
        
    def get_rating_error(self, r, p, q, b_u, b_i):
        return r- (self.myu + b_i + b_u + numpy.dot(p, q))

    def get_error(self, beta):
        """
        目的関数
        @param(beta) 正規化係数
        """
        self.error = 0.0

        for user in range(len(self.R)):
            for item in xrange(len(self.R[user])):
                if R[user][item] == 0:
                    continue
                self.error += pow(self.get_rating_error(self.R[user][item], self.P[:,user], self.Q[:,item], self.B_u[user], self.B_i[item]), 2)
        # 正規化項
        self.error += beta * (numpy.linalg.norm(self.B_u) + numpy.linalg.norm(self.B_i) + numpy.linalg.norm(self.P) + numpy.linalg.norm(self.Q))
    
    def learning(self, K, steps = 5000, gamma = 0.0002, beta = 0.02, threshold = 0.001):
        """
        ユーザー行列Pとアイテム行列Qを最適化によって求める
        @param(K) 疑似行列の次元
        @param(steps) 学習回数
        @param(alpha) 学習係数
        @param(beta) 正規化係数
        @param(threshold) 誤差の閾値
        """
        self.P = numpy.random.rand(K, len(self.R)) # ユーザー嗜好ベクトル
        self.Q = numpy.random.rand(K, len(self.R[0])) # アイテム嗜好ベクトル
        self.B_u = numpy.random.rand(len(self.R)) # ユーザーバイアス
        self.B_i = numpy.random.rand(len(self.R[0])) # アイテムバイアス

        for step in xrange(steps):
            for user in xrange(len(self.R)):
                for item in xrange(len(self.R[user])):
                    if self.R[user][item] == 0:
                        continue
                    err = self.get_rating_error(self.R[user][item], self.P[:,user], self.Q[:,item], self.B_u[user], self.B_i[item])
                    for k in xrange(K):
                        self.P[k][user] += gamma * (err*self.Q[k][item] - beta*self.P[k][user])
                        self.Q[k][item] += gamma * (err*self.P[k][user] - beta*self.Q[k][item])
                        self.B_u[user] += gamma * (err - beta*self.B_u[user])
                        self.B_i[item] += gamma * (err - beta+self.B_i[item])
            self.get_error(beta)
            if self.error < threshold:
                self.nR = numpy.dot(self.P.T, self.Q) # 得られた評価値行列
                break
