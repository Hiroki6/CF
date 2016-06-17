# -*- coding:utf-8 -*-
"""
Factorization Machineをcythonを使って高速化
学習手法は確率的勾配法(SGD)
学習率はAdaGradを使用
正規化項は「Learning Recommender Systems with Adaptive Regularization」参考
"""

from libc.math cimport pow, sqrt
import math
import random
import numpy as np
cimport numpy as np
cimport cython

np.import_array()

ctypedef np.float64_t DOUBLE
ctypedef np.int64_t INTEGER

cdef class CyFmSgd:
    """
    parameters
    R : 学習データ配列(FMフォーマット形式) N * n
    R_v : テスト用データ配列(FMフォーマット形式) regsとgradsの最適化用
    targets : 学習データの教師ラベル N
    regs_targets : テストデータの教師ラベル len(R_v)
    w_0 : バイアス 1
    W : 各特徴量の重み n
    V : 各特徴量の相互作用の重み n * K
    E : 各データの予測誤差 N
    adagrad_w_0 : adagradにおけるw_0の保存配列 1
    adagrad_V : adagradにおけるVの保存配列 n * K
    adagrad_W : adagradにおけるWの保存配列 n
    N : 学習データ数
    n : 特徴量の総数
    l_rate : 学習率
    r_param : 勾配率
    K : Vの次元
    step : 学習ステップ数
    regs : regulations 配列 K+2 (0: w_0, 1: W, 2~K+2: V)
    """

    cdef:
        np.ndarray R
        np.ndarray R_v
        np.ndarray targets
        np.ndarray regs_targets
        np.ndarray W
        np.ndarray V
        np.ndarray E
        np.ndarray adagrad_V
        np.ndarray adagrad_W
        double adagrad_w_0
        double w_0
        long n
        long N
        long N_v
        np.ndarray regs
        double l_rate
        int K
        int step
        double error

    def __cinit__(self,
                    np.ndarray[DOUBLE, ndim=2, mode="c"] R,
                    np.ndarray[DOUBLE, ndim=2, mode="c"] R_v,
                    np.ndarray[INTEGER, ndim=1, mode="c"] targets,
                    np.ndarray[INTEGER, ndim=1, mode="c"] regs_targets,
                    np.ndarray[DOUBLE, ndim=1, mode="c"] W,
                    np.ndarray[DOUBLE, ndim=2, mode="c"] V,
                    double w_0,
                    long n,
                    long N,
                    long N_v,
                    np.ndarray[DOUBLE, ndim=1, mode="c"] E,
                    np.ndarray[DOUBLE, ndim=1, mode="c"] regs,
                    double l_rate,
                    int K,
                    int step):
        self.R = R
        self.R_v = R_v
        self.targets = targets
        self.regs_targets = regs_targets
        self.W = W
        self.V = V
        self.w_0 = w_0
        self.n = n
        self.N = N
        self.N_v = N_v
        self.E = E
        self.regs = regs
        self.l_rate = l_rate
        self.K = K
        self.step = step

    def get_sum_error(self):

        cdef:
            long data_index
            int f

        self.error = np.sum(self.E**2)

        self.error += self.regs[0] * pow(self.w_0, 2) + self.regs[1] * np.sum(self.W**2)
        for f in xrange(self.K):
            self.error += self.regs[f+2] * np.sum(np.transpose(self.V)[f]**2)

    def get_all_error(self):
        """
        全ての学習データの誤差を計算
        """
        cdef:
            long data_index

        for data_index in xrange(self.N):
            print data_index
            self._get_error(data_index)

    cdef void _get_error(self, long data_index):
        """
        誤差計算
        """
        cdef:
            double features = 0.0
            double iterations = 0.0
            int f

        features = np.dot(self.W, self.R[data_index])
        for f in xrange(self.K):
            iterations += pow(np.dot(self.V[:,f], self.R[data_index]), 2) - np.dot(self.V[:,f]**2, self.R[data_index]**2)

        self.E[data_index] = (self.w_0 + features + iterations/2) - self.targets[data_index]

    cdef void _update_w_0(self, long data_index):
        """
        w_0の更新
        """
        cdef:
            double grad_value = 0.0
            double update_value = 0.0

        grad_value = 2 * self.l_rate*(self.E[data_index] + self.regs[0]*self.w_0)
 
        self.adagrad_w_0 += grad_value * grad_value
        update_value = self.l_rate * grad_value / sqrt(self.adagrad_w_0)
        self.w_0 -= update_value
        #self.E[data_index] -= update_value

    cdef void _update_W(self, long data_index, long i):
        """
        W[i]の更新
        """
        cdef:
            double grad_value = 0.0
            double update_value = 0.0

        grad_value = 2 * (self.E[data_index]*self.R[data_index][i] + self.regs[1]*self.W[i])
        self.adagrad_W[i] += grad_value * grad_value
        update_value = self.l_rate * grad_value / sqrt(self.adagrad_W[i])
        self.W[i] -= update_value
        #self.E[data_index] -= update_value

    cdef void _update_V(self, long data_index, long i, int f):
        """
        V[i][f]の更新
        """
        cdef:
            double grad_value = 0.0
            double updata_value = 0.0
            double h = 0.0
        
        h = np.dot(np.transpose(self.V)[f], self.R[data_index]) - self.V[i][f]*self.R[data_index][i]
        h *= self.R[data_index][i]
        grad_value = 2 * (self.E[data_index]*h + self.regs[f+2]*self.V[i][f])
        self.adagrad_V[i][f] += grad_value * grad_value
        update_value = self.l_rate * grad_value / sqrt(self.adagrad_V[i][f])
        self.V[i][f] -= update_value
        #self.E[data_index] -= update_value

    def repeat_optimization(self):
 
        cdef:
            long i
            int f
            long data_index
            bint nan_flag = False
            double pre_w_0
            np.ndarray[DOUBLE, ndim=1, mode="c"] pre_W
            np.ndarray[DOUBLE, ndim=2, mode="c"] pre_V
       
        for data_index in xrange(self.N):
            """
            パラメータの最適化
            """
            pre_w_0 = self.w_0
            pre_W = self.W
            pre_V = self.V
            if nan_flag:
                break
            print "data_index %d" % data_index
            self._update_w_0(data_index)
            for i in xrange(self.n):
                if self.R[data_index][i] <= 0:
                    continue
                self._update_W(data_index, i)
                for f in xrange(self.K):
                    self._update_V(data_index, i, f)
                    if math.isnan(self.V[i][f]):
                        nan_flag = True
                        break
            self.calc_regs(pre_w_0, pre_W, pre_V)

    cdef void calc_regs(self, double pre_w_0, np.ndarray[DOUBLE, ndim=1, mode="c"] pre_W, np.ndarray[DOUBLE, ndim=2, mode="c"] pre_V):
        """
        regsの最適化
        """
        cdef:
            double new_r
            double err
            int f
            long random_index
        
        random_index = random.randint(0, self.N_v-1)
        err = 2 * self.calc_error(random_index)
        new_r = self.regs[0] - self.l_rate * (err * -2 * self.l_rate * pre_w_0)
        self.regs[0] = new_r if new_r >= 0 else 0
        new_r = self.regs[1] - self.l_rate * (err * -2 * self.l_rate * np.dot(pre_W, self.R_v[random_index]))
        self.regs[1] = new_r if new_r >= 0 else 0
        for f in xrange(self.K):
            new_r = self.regs[f+2] - self.l_rate * (err * -2 * self.l_rate * (np.dot(self.R_v[random_index], np.transpose(self.V)[f]) * np.dot(self.R_v[random_index], np.transpose(pre_V)[f]) - np.sum((self.R_v[random_index]**2)*np.transpose(self.V)[f]*np.transpose(pre_V)[f])))
            self.regs[f+2] = new_r if new_r >= 0 else 0

    cdef double calc_error(self, long data_index):

        cdef:
            double features = 0.0
            double iterations = 0.0
            int f

        features = np.dot(self.W, self.R_v[data_index])
        for f in xrange(self.K):
            iterations += pow(np.dot(self.V[:,f], self.R_v[data_index]), 2) - np.dot(self.V[:,f]**2, self.R_v[data_index]**2)

        return (self.w_0 + features + iterations/2) - self.regs_targets[data_index]

    def learning(self):

        cdef:
            int s

        self.adagrad_w_0 = 0.0
        self.adagrad_W = np.zeros(self.n)
        self.adagrad_V = np.zeros((self.n, self.K))
        self.get_all_error()
        for s in xrange(self.step):
            print "Step %d" % s
            self.repeat_optimization()
            self.get_all_error()
            self.get_sum_error()
            if self.error <= 100:
                break

    cdef double _calc_rating(self,
                    np.ndarray[DOUBLE, ndim=1, mode="c"] test_matrix):

        cdef:
            # 各特徴量の重み
            double features = 0.0
            # 相互作用の重み
            double iterations = 0.0
            int f

        features = np.dot(self.W, test_matrix)
        for f in xrange(self.K):
            iterations += pow(np.dot(self.V[:,f], test_matrix), 2) - np.dot(self.V[:,f]**2, test_matrix**2)
        return self.w_0 + features + iterations/2

    def predict(self, test_matrix):
        return self._calc_rating(test_matrix)

    def get_errors(self):
        return self.E

    def get_w_0(self):
        return self.w_0

    def get_w(self):
        return self.W

    def get_v(self):
        return self.V
    
    def get_self_error(self):
        return self.error
