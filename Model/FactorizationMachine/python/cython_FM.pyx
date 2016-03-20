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

    cdef np.ndarray R
    cdef np.ndarray targets
    cdef long n
    cdef long N
    cdef int K
    cdef int step
    cdef double w_0
    cdef double beta
    cdef np.ndarray W
    cdef np.ndarray V
    cdef np.ndarray E
    cdef np.ndarray Q

    def __cinit__(self,
                    np.ndarray[DOUBLE, ndim=2, mode="c"] R,
                    np.ndarray[INTEGER, ndim=1, mode="c"] targets,
                    np.ndarray[DOUBLE, ndim=1, mode="c"] W,
                    np.ndarray[DOUBLE, ndim=2, mode="c"] V,
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
        self.w_0 = w_0
        self.beta = beta
        self.n = n
        self.N = N
        self.K = K
        self.step = step

    cdef get_all_error(self):
        
        cdef long N = self.N
        cdef np.ndarray[INTEGER, ndim=1, mode="c"] targets = self.targets
        for data_index in xrange(N):
            print data_index
            self.get_error(data_index, targets[data_index])

    cdef get_q_error(self, long data_index):

        cdef np.ndarray[DOUBLE, ndim=2, mode="c"] Q = self.Q
        cdef np.ndarray[DOUBLE, ndim=2, mode="c"] V = self.V
        cdef np.ndarray[DOUBLE, ndim=2, mode="c"] R = self.R
        for f in xrange(len(V[0])):
            Q[data_index][f] = np.dot(V[:,f], R[data_index])

        self.Q = Q

    cdef get_error(self, long data_index, int target):
        
        # 各特徴量の重み
        cdef double features = 0.0
        # 相互作用の重み
        cdef double iterations = 0.0
        cdef double w_0 = self.w_0
        cdef int K = self.K
        cdef np.ndarray[DOUBLE, ndim=2, mode="c"] V = self.V
        cdef np.ndarray[DOUBLE, ndim=2, mode="c"] R = self.R
        cdef np.ndarray[DOUBLE, ndim=1, mode="c"] W = self.W
        cdef np.ndarray[DOUBLE, ndim=1, mode="c"] E = self.E

        for f in xrange(K):
            iterations += pow(np.dot(V[:,f], R[data_index]), 2) - np.dot(V[:,f]**2, R[data_index]**2)
        E[data_index] = (w_0 + features + iterations) - target
        self.E = E

    """
    w_0の更新
    """
    cdef update_global_bias(self):

        cdef double error_sum = 0.0
        cdef double w_0 = self.w_0
        cdef np.ndarray[DOUBLE, ndim=1, mode="c"] E = self.E
        cdef double new_w0 = 0.0
        cdef long N = self.N
        cdef double beta = self.beta

        error_sum = np.sum(E) - w_0*N

        new_w0 = -(error_sum) / (N + beta*error_sum)
        for data_index in xrange(N):
            E[data_index] += new_w0 - w_0

        self.w_0 = new_w0
        self.E = E
    
    """
    Wの更新
    """
    cdef update_weight(self):
   
        cdef double error_sum = 0.0
        cdef double feature_square_sum = 0.0
        cdef long n = self.n
        cdef long N = self.N
        cdef double beta = self.beta
        cdef double new_wl = 0.0
        #cdef np.ndarray[DOUBLE, ndim=2, mode="c"] R = self.R
        cdef np.ndarray R = self.R
        #cdef np.ndarray[DOUBLE, ndim=1, mode="c"] W = self.W
        cdef np.ndarray W = self.W
        #cdef np.ndarray[DOUBLE, ndim=1, mode="c"] E = self.E
        cdef np.ndarray E = self.E

        for l in xrange(n):
            error_sum = 0.0
            fetaure_square_sum = 0.0
            error_sum = sum((E[data_index] - W[l] * R[data_index][l]) * R[data_index][l] for data_index in xrange(N))
            feature_square_sum = np.sum(R[:,l] ** 2)
            new_wl = -(error_sum) / (feature_square_sum + beta*error_sum)
            for data_index in xrange(N):
                E[data_index] += (new_wl - W[l]) * R[data_index][l]
            self.W[l] = new_wl
        self.E = E

    """
    Vの更新
    """
    cdef update_interaction(self):
        
        cdef double error_sum = 0.0
        cdef double new_v = 0.0
        cdef double h_square_sum = 0.0
        cdef double h_v = 0.0
        cdef long n = self.n
        cdef long N = self.N
        cdef int K = self.K
        cdef double beta = self.beta
        cdef np.ndarray[DOUBLE, ndim=2, mode="c"] V = self.V
        cdef np.ndarray[DOUBLE, ndim=2, mode="c"] R = self.R
        cdef np.ndarray[DOUBLE, ndim=1, mode="c"] W = self.W
        cdef np.ndarray[DOUBLE, ndim=1, mode="c"] E = self.E
        cdef np.ndarray[DOUBLE, ndim=2, mode="c"] Q = self.Q

        for f in xrange(K):
            for l in xrange(n):
                error_sum = 0.0
                h_square_sum = 0.0
                for data_index in xrange(N):
                    h_v = (R[data_index][l] * Q[data_index]) - (pow(R[data_index][l], 2)*V[l][f])
                    error_sum += (E[data_index] - V[l][f] * h_v) * h_v
                    h_square_sum += pow(h_v, 2)
                new_v = -(error_sum) / (h_square_sum + beta*error_sum)
                for data_index in xrange(N):
                    E[data_index] += (new_v - V[l][f]) * R[data_index][l]
                    Q[data_index] += (new_v - V[l][f]) * R[data_index][l]
                self.V[l][f] = new_v

        self.E = E
        self.Q = W

    cdef repeat_optimization(self):

        self.update_global_bias()
        self.update_weight()
        self.update_interaction()

    """
    ALSの学習
    """
    def learning(self):
        
        self.E = np.zeros(self.N)
        self.Q = np.zeros((self.N, self.K))

        """
        誤差の計算
        """
        self.get_all_error()
        for i in xrange(self.step):
            print i
            self.repeat_optimization()
