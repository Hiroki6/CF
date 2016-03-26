# encoding: utf-8
#cython: boundscheck=False
#cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
cimport cython
import time

ctypedef np.float64_t DOUBLE_t
ctypedef np.int_t INT_t
"""
Matrix Factorizationをcythonを用いて高速化する
"""
cdef class fastMF(object):
    
    cdef:
        np.ndarray P
        np.ndarray Q
        np.ndarray nR
        np.ndarray R
        long u_num
        long i_num
        int K
        int steps
        double gamma
        double beta
        double threshold

    def __cinit__(self,
            np.ndarray[DOUBLE_t, ndim=2, mode = 'c'] R,
            np.ndarray[DOUBLE_t, ndim=2, mode = 'c'] P,
            np.ndarray[DOUBLE_t, ndim=2, mode = 'c'] Q,
            long u_num,
            long i_num,
            int K,
            int steps,
            double gamma,
            double beta,
            double threshold):
        self.R = R
        self.P = P
        self.Q = Q
        self.K = K
        self.u_num = u_num
        self.i_num = i_num
        self.steps = steps
        self.gamma = gamma
        self.beta = beta
        self.threshold = threshold

    cdef double get_rating_error(self, int user, int item):
        return self.R[user][item] - np.dot(self.P[:,user], self.Q[:,item])

    cdef double get_error(self):

        cdef:
            double error = 0.0
            long user_index
            long item_index
            
        for user_index in xrange(self.u_num):
            for item_index in xrange(self.i_num):
                if self.R[user_index][item_index] == 0:
                    continue
                error += pow(self.get_rating_error(user_index, item_index), 2)

        error += self.beta * (np.linalg.norm(self.P) + np.linalg.norm(self.Q))
        return error

    def learning(self):

        cdef:
            double err = 0.0
            int step
            long user_index
            long item_index
            int k
            double all_error = 0.0

        for step in xrange(self.steps):
            for user_index in xrange(self.u_num):
                for item_index in xrange(self.i_num):
                    if self.R[user_index][item_index] == 0:
                        continue
                    err = self.get_rating_error(user_index, item_index)
                    for k in xrange(self.K):
                        self.P[k][user_index] += self.gamma * (err*self.Q[k][item_index] - self.beta*self.P[k][user_index])
                        self.Q[k][item_index] += self.gamma * (err*self.P[k][user_index] - self.beta*self.Q[k][item_index])
            
            all_error = self.get_error()
            print all_error
            if all_error < self.threshold:
                self.nR = np.dot(np.transpose(self.P), self.Q) # 得られた評価値行列
                return
        
        self.nR = np.dot(np.transpose(self.P), self.Q)

    cpdef double predict(self, int user, int item):
        return self.nR[user][item]
