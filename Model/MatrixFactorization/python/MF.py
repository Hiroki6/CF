# -*- coding:utf-8 -*-
"""
SVD++の実装
参考文献:
    「Advances in Collaborative Filtering」,Y.Koran and R.Bell ,2007 IEEE
R: ユーザーとアイテムをインデックスにもつ二次元ディクショナリ
"""
import numpy as np
import cythonMF as cMF
import time
import sys
sys.dont_write_bytecode = True 

class basicMF:
    def __init__(self, R):
        self.R = R # 評価値行列
        self.u_num = len(R)
        self.i_num = len(R[0])
    
    def get_rating_error(self, user, item):
        return self.R[user][item] - np.dot(self.P[:,user], self.Q[:,item])
    
    def get_error(self, beta):
        #目的関数
        #@param(beta) 正規化係数
        self.error = 0.0

        for user_index in xrange(self.u_num):
            for item_index in xrange(self.i_num):
                if self.R[user_index][item_index] == 0:
                    continue
                self.error += pow(self.get_rating_error(user_index, item_index), 2)
        # 正規化項
        self.error += beta * (np.linalg.norm(self.P) + np.linalg.norm(self.Q))
    
    def learning(self, K, steps = 30, gamma = 0.005, beta = 0.02, threshold = 0.1):
        """
        ユーザー行列Pとアイテム行列Qを最適化によって求める
        @param(K) 疑似行列の次元
        @param(steps) 学習回数
        @param(alpha) 学習係数
        @param(beta) 正規化係数
        @param(threshold) 誤差の閾値
        """
        self.P = np.random.rand(K, self.u_num)
        self.Q = np.random.rand(K, self.i_num)

        for step in xrange(steps):
            start = time.time()
            for user_index in xrange(self.u_num):
                for item_index in xrange(self.i_num):
                    if self.R[user_index][item_index] == 0:
                        continue
                    err = self.get_rating_error(user_index, item_index)
                    for k in xrange(K):
                        self.P[k][user_index] += gamma * (err*self.Q[k][item_index] - beta*self.P[k][user_index])
                        self.Q[k][item_index] += gamma * (err*self.P[k][user_index] - beta*self.Q[k][item_index])
            elapsed_time = time.time() - start
            print elapsed_time
            self.get_error(beta)
            print self.error
            if self.error < threshold:
                self.nR = np.dot(self.P.T, self.Q) # 得られた評価値行列
                return 

        self.nR = np.dot(self.P.T, self.Q)

    def predict(self, user, item):
        return self.nR[user][item]

    def recommends(self, user):
        rankings = [(self.nR[user][item], str(item)) for item in self.R[user] if self.R[user][item] == 0]
        rankings.sort()
        return rankings

"""
cython化したMF
"""
class Cy_basicMF:
    
    def __init__(self, R):
        self.R = R # 評価値行列
        self.u_num = len(R)
        self.i_num = len(R[0])

    def learning(self, K, steps = 30, gamma = 0.005, beta = 0.02, threshold = 0.1):

        self.P = np.random.rand(K, self.u_num)
        self.Q = np.random.rand(K, self.i_num)
        self.cython_obj = cMF.fastMF(self.R, self.P, self.Q, self.u_num, self.i_num, K, steps, gamma, beta, threshold)
        self.cython_obj.learning()

    def predict(self, user, item):
        return self.cython_obj.predict(user, item)

class svd:

    def __init__(self, R):
        self.R = R # 評価値行列
        self.u_num = len(R)
        self.i_num = len(R[0])
        self.myu = self.get_ave()

    def get_ave(self):

        count = 0
        sumRate = 0
        for user_index in xrange(self.u_num):
            for item_index in xrange(self.i_num):
                if not self.R[user_index][item_index] == 0:
                    count += 1
                    sumRate += self.R[user_index][item_index]

        if count == 0:
            return 0
        else:
            return sumRate/count
        
    def get_rating_error(self, user, item):
        return self.R[user][item] - (self.myu + self.B_u[user] + self.B_i[item] + np.dot(self.P[:, user], self.Q[:, item]))

    def get_error(self, beta):
        """
        目的関数
        @param(beta) 正規化係数
        """
        self.error = 0.0

        for user_index in xrange(self.u_num):
            for item_index in xrange(self.i_num):
                if self.R[user_index][item_index] == 0:
                    continue
                self.error += pow(self.get_rating_error(user_index, item_index), 2)
        # 正規化項
        self.error += beta * (np.linalg.norm(self.B_u) + np.linalg.norm(self.B_i) + np.linalg.norm(self.P) + np.linalg.norm(self.Q))
    
    def learning(self, K, steps = 30, gamma = 0.005, beta = 0.02, threshold = 0.01):
        """
        ユーザー行列Pとアイテム行列Qを最適化によって求める
        @param(K) 疑似行列の次元
        @param(steps) 学習回数
        @param(alpha) 学習係数
        @param(beta) 正規化係数
        @param(threshold) 誤差の閾値
        """
        self.P = np.random.rand(K, self.u_num) # ユーザー嗜好ベクトル
        self.Q = np.random.rand(K, self.i_num) # アイテム嗜好ベクトル
        self.B_u = np.random.rand(self.u_num) # ユーザーバイアス
        self.B_i = np.random.rand(self.i_num) # アイテムバイアス

        for step in xrange(steps):
            for user_index in xrange(self.u_num):
                for item_index in xrange(self.i_num):
                    if self.R[user_index][item_index] == 0:
                        continue
                    err = self.get_rating_error(user_index, item_index)
                    for k in xrange(K):
                        self.P[k][user_index] += gamma * (err*self.Q[k][item_index] - beta*self.P[k][user_index])
                        self.Q[k][item_index] += gamma * (err*self.P[k][user_index] - beta*self.Q[k][item_index])
                        self.B_u[user_index] += gamma * (err - beta*self.B_u[user_index])
                        self.B_i[item_index] += gamma * (err - beta*self.B_i[item_index])
            self.get_error(beta)
            print self.error
            if self.error < threshold:
                self.nR = np.dot(self.P.T, self.Q) # 得られた評価値行列
                return

        self.nR = np.dot(self.P.T, self.Q)

    def predict(self, user, item):
        return self.nR[user][item] + self.myu + self.B_u[user] + self.B_i[item]

class svdPlus(svd):
    
    def get_rating_error(self, user, item):

        num_rate, sum_rate = self.get_sum_rate_by_user(user)

        return self.R[user][item] - (self.myu + self.B_u[user] + self.B_i[item] + np.dot((self.P[:, user] + (sum_rate)/sqrt(num_rate)), self.Q[:, item]))

    def get_sum_rate_by_user(self, user):
        """
        ユーザーが評価したアイテムの評価値の合計を返す
        """
        sum_rate = np.array(len(self.P))
        count = 0
        for item in self.R[user]:
            count += 1
            sum_rate = sum_rate + self.Y[:, item]

        return count, sum_rate
