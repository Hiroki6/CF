# encoding: utf-8
#cython: boundscheck=False
#cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DOUBLE_t
#ctypedef np.int_t INT_t
"""
Matrix Factorizationをcythonを用いて高速化する
"""
cdef class fastMF(object):

    cdef np.ndarray P
    cdef np.ndarray Q
    cdef np.ndarray nR
    cdef double error
    def __cinit__(self, R, np.ndarray[DOUBLE_t, ndim=2, mode = 'c'] P, np.ndarray[DOUBLE_t, ndim=2, mode = 'c'] Q):
        self.R = R
        self.P = P
        self.Q = Q

    cdef get_rating_error(self, int user, int item):
        
        cdef np.ndarray[DOUBLE_t, ndim=2, mode="c"] P = self.P
        cdef np.ndarray[DOUBLE_t, ndim=2, mode="c"] Q = self.Q
        return self.R[user][item] - np.dot(P[:,user], Q[:,item])

    cdef get_error(self, double beta):

        self.error = 0.0
        cdef np.ndarray[DOUBLE_t, ndim=2, mode="c"] P = self.P
        cdef np.ndarray[DOUBLE_t, ndim=2, mode="c"] Q = self.Q

        for user_index in self.R:
            for item_index in self.R[user_index]:
                if self.R[user_index][item_index] == 0:
                    continue
                self.error += pow(self.get_rating_error(user_index, item_index), 2)

        self.error += beta * (np.linalg.norm(P) + np.linalg.norm(Q))

    cdef learning(self, int K, int steps = 30, double gamma = 0.005, double beta = 0.02, threshold = 0.1):
        cdef double err = 0.0
        cdef np.ndarray[DOUBLE_t, ndim=2, mode="c"] P = self.P
        cdef np.ndarray[DOUBLE_t, ndim=2, mode="c"] Q = self.Q

        for step in xrange(steps):
            for user_index in self.R:
                for item_index in self.R[user_index]:
                    if self.R[user_index][item_index] == 0:
                        continue
                    err = self.get_rating_error(user_index, item_index)
                    for k in xrange(K):
                        P[k][user_index] += gamma * (err*Q[k][item_index] - beta*P[k][user_index])
                        Q[k][item_index] += gamma * (err*P[k][user_index] - beta*Q[k][item_index])
                    self.P = P
                    self.Q = Q

            self.get_error(beta)
            print self.error
            if self.error > threshold:
                self.nR = np.dot(np.transpose(P), Q) # 得られた評価値行列
                return
        
        
        self.nR = np.dot(np.transpose(P), Q)

    cdef predict(self, int user, int item):
        
        cdef np.ndarray[DOUBLE_t, ndim=2, mode="c"] nR = self.nR
        return nR[user][item]
