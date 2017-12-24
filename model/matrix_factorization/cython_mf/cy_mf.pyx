# encoding: utf-8
#cython: boundscheck=False
#cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange
import time
from libc.math cimport pow

ctypedef np.float64_t DOUBLE_t
ctypedef np.int_t INT_t
"""
Matrix Factorizationをcythonを用いて高速化する
"""
cdef class FastMF(object):
    
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

    cdef inline double _get_rating_error(self, int user, int item):
        return self.R[user][item] - np.dot(self.P[:,user], self.Q[:,item])

    cdef double _get_error(self):

        cdef:
            double error = 0.0
            long user_index
            long item_index
            np.ndarray[DOUBLE_t, ndim=1, mode = 'c'] user_matrix
            double rating

        for user_index, user_matrix in enumerate(self.R):
            for item_index, rating in enumerate(user_matrix):
                if not rating:
                    continue
                error += pow(self._get_rating_error(user_index, item_index), 2)

        print(error)
        error += self.beta / 2 * (np.linalg.norm(self.P) + np.linalg.norm(self.Q))
        return error

    def learning(self):

        cdef:
            double err = 0.0
            int step
            long user_index
            long item_index
            double start
            int k
            double all_error = 0.0
            np.ndarray[DOUBLE_t, ndim=1, mode = 'c'] user_matrix
            double rating

        for step in xrange(self.steps):
            for user_index, user_matrix in enumerate(self.R):
                for item_index, rating in enumerate(user_matrix):
                    if not rating:
                        continue
                    pre_u = np.transpose(self.P)[user_index]
                    err = self._get_rating_error(user_index, item_index)
                    np.transpose(self.P)[user_index] += self.gamma * (2 * err * np.transpose(self.Q)[item_index] - self.beta * np.transpose(self.P)[user_index])
                    np.transpose(self.Q)[item_index] += self.gamma * (2 * err * pre_u - self.beta * np.transpose(self.Q)[item_index])
           
            all_error = self._get_error()
            print all_error
            if all_error < self.threshold:
                self.nR = np.dot(np.transpose(self.P), self.Q) # 得られた評価値行列
                return
        
        self.nR = np.dot(np.transpose(self.P), self.Q)

    cpdef double predict(self, int user, int item):
        return self.nR[user][item]
