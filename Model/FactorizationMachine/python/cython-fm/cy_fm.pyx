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

    cdef get_all_error(self):
        
        cdef:
            long data_index
            
        for data_index in xrange(self.N):
            print data_index
            self.E[data_index] = self.get_error(data_index, self.targets[data_index])
            self.Q[data_index] = self.get_q_error(self.Q[data_index], data_index)

    cdef np.ndarray get_q_error(self, np.ndarray[DOUBLE, ndim=1, mode="c"] q, long data_index):
        
        cdef:
            int f
        for f in xrange(self.K):
            q[f] = np.dot(self.V[:,f], self.R[data_index])

        return q

    cdef double get_error(self, long data_index, int target):
        
        cdef:
        # 各特徴量の重み
            double features = 0.0
        # 相互作用の重み
            double iterations = 0.0
            int f

        for f in xrange(self.K):
            iterations += pow(np.dot(self.V[:,f], self.R[data_index]), 2) - np.dot(self.V[:,f]**2, self.R[data_index]**2)
        # 型付け
        return (self.w_0 + features + iterations) - target

    """
    w_0の更新
    """
    cdef update_global_bias(self):
        
        cdef:
            double error_sum = 0.0
            double new_w0 = 0.0
            long data_index

        error_sum = np.sum(self.E) - self.w_0*self.N

        new_w0 = -(error_sum) / (self.N + self.beta*error_sum)
        for data_index in xrange(self.N):
            self.E[data_index] += new_w0 - self.w_0
    
    """
    Wの更新
    """
    cdef update_weight(self):
        
        cdef:
            double error_sum = 0.0
            double feature_square_sum = 0.0
            double new_wl = 0.0
            long l
            long data_index

        for l in xrange(self.n):
            print l
            error_sum = 0.0
            fetaure_square_sum = 0.0
            error_sum = sum((self.E[data_index] - self.W[l] * self.R[data_index][l]) * self.R[data_index][l] for data_index in xrange(self.N))
            feature_square_sum = np.sum(self.R[:,l] ** 2)
            new_wl = -(error_sum) / (feature_square_sum + self.beta*error_sum)
            for data_index in xrange(self.N):
                self.E[data_index] += (new_wl - self.W[l]) * self.R[data_index][l]
            self.W[l] = new_wl

    """
    Vの更新
    """
    cdef update_interaction(self):
       
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
                for data_index in xrange(self.N):
                    self.E[data_index] += (new_v - self.V[l][f]) * self.R[data_index][l]
                    self.Q[data_index][f] += (new_v - self.V[l][f]) * self.R[data_index][l]
                self.V[l][f] = new_v

    cdef repeat_optimization(self):

        print "w_0更新"
        self.update_global_bias()
        print "W更新"
        self.update_weight()
        print "V更新"
        self.update_interaction()

    """
    ALSの学習
    """
    def learning(self):

        """
        誤差の計算
        """
        self.get_all_error()
        for i in xrange(self.step):
            print i
            self.repeat_optimization()
