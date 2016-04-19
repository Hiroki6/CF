# -*- coding:utf-8 -*-
"""
Factorization Machineをcythonを使って高速化
"""

from libc.math cimport pow
import numpy as np
cimport numpy as np
cimport cython

np.import_array()

ctypedef np.float64_t DOUBLE
ctypedef np.int64_t INTEGER

cdef class cy_FM:

    cdef:
        np.ndarray R
        np.ndarray targets
        long n
        long N
        int K
        int step
        double w_0
        double beta
        np.ndarray W
        np.ndarray V
        np.ndarray E
        np.ndarray Q

    def __cinit__(self,
                    np.ndarray[DOUBLE, ndim=2, mode="c"] R,
                    np.ndarray[INTEGER, ndim=1, mode="c"] targets,
                    np.ndarray[DOUBLE, ndim=1, mode="c"] W,
                    np.ndarray[DOUBLE, ndim=2, mode="c"] V,
                    np.ndarray[DOUBLE, ndim=1, mode="c"] E,
                    np.ndarray[DOUBLE, ndim=2, mode="c"] Q,
                    double w_0,
                    double beta,
                    long n,
                    long N,
                    int K,
                    int step):
        self.R = R
        self.targets = targets
        self.n = len(self.R[0])
        self.N = len(self.R)
        self.W = W
        self.V = V
        self.E = E
        self.Q = Q
        self.w_0 = w_0
        self.beta = beta
        self.n = n
        self.N = N
        self.K = K
        self.step = step

    cdef _get_all_error(self):
        
        cdef:
            long data_index
            
        for data_index in xrange(self.N):
            print data_index
            self._get_error(data_index)
            self._get_q_error(data_index)

    cdef _get_q_error(self, long data_index):
        
        cdef:
            int f
        for f in xrange(self.K):
            self.Q[data_index][f] = np.dot(self.V[:,f], self.R[data_index])

    cdef _get_error(self, long data_index):
        
        cdef:
            # 各特徴量の重み
            double features = 0.0
            # 相互作用の重み
            double iterations = 0.0
            int f

        features = np.dot(self.W, self.R[data_index])
        for f in xrange(self.K):
            iterations += pow(np.dot(self.V[:,f], self.R[data_index]), 2) - np.dot(self.V[:,f]**2, self.R[data_index]**2)
        # 型付け
        self.E[data_index] = (self.w_0 + features + iterations/2) - self.targets[data_index]

    cdef double _print_sum_error(self):
        print np.sum(self.E)

    """
    w_0の更新
    """
    cdef _update_global_bias(self):
        
        cdef:
            double error_sum = 0.0
            double new_w0 = 0.0
            long data_index

        error_sum = np.sum(self.E) - self.w_0*self.N

        new_w0 = -(error_sum) / (self.N + self.beta*error_sum)
        for data_index in xrange(self.N):
            self.E[data_index] += new_w0 - self.w_0

        self.w_0 = new_w0
    
    """
    Wの更新
    """
    cdef _update_weight(self):
        
        cdef:
            double error_sum = 0.0
            double feature_square_sum = 0.0
            double new_wl = 0.0
            long l
            long data_index

        for l in xrange(self.n):
            print l
            error_sum = np.sum((self.E - self.W[l] * np.transpose(self.R)[l]) * np.transpose(self.R)[l])
            feature_square_sum = np.sum(self.R[:,l] ** 2)
            new_wl = -(error_sum) / (feature_square_sum + self.beta*error_sum)
            self.E += (new_wl - self.W[l]) * np.transpose(self.R)[l]
            self.W[l] = new_wl

    """
    Vの更新
    """
    cdef _update_interaction(self):
       
        cdef:
            double error_sum = 0.0
            double new_v = 0.0
            double h_square_sum = 0.0
            double h_v = 0.0
            long f
            long l
            long data_index
    
        for f in xrange(self.K):
            print "f %d" % (f)
            for l in xrange(self.n):
                print "l %d" % (l)
                error_sum = 0.0
                h_square_sum = 0.0
                for data_index in xrange(self.N):
                    h_v = (self.R[data_index][l] * self.Q[data_index][f]) - (pow(self.R[data_index][l], 2)*self.V[l][f])
                    error_sum += (self.E[data_index] - self.V[l][f] * h_v) * h_v
                    h_square_sum += pow(h_v, 2)
                new_v = -(error_sum) / (h_square_sum + self.beta*error_sum)
                self.E += (new_v - self.V[l][f]) * np.transpose(self.R)[l]
                np.transpose(self.Q)[f] += (new_v - self.V[l][f]) * np.transpose(self.R)[l]
                self.V[l][f] = new_v

    cdef _repeat_optimization(self):

        print "w_0更新"
        self._update_global_bias()
        print "W更新"
        self._update_weight()
        print "V更新"
        self._update_interaction()

    """
    ALSの学習
    """
    def learning(self):

        """
        誤差の計算
        """
        self._get_all_error()
        self._print_sum_error()
        for i in xrange(self.step):
            print "Learning Iteration %d" % i
            self._repeat_optimization()

    """
    test_matrixに対する予測値の算出
    """
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
