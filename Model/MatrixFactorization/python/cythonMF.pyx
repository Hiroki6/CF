# encoding: utf-8
#cython: boundscheck=False
#cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
cimport cython

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
        dict R
        double error
        int steps
        double gamma
        double beta
        double threshold

    def __cinit__(self,
            dict R,
            np.ndarray[DOUBLE_t, ndim=2, mode = 'c'] P,
            np.ndarray[DOUBLE_t, ndim=2, mode = 'c'] Q,
            double error,
            int steps,
            double gamma,
            double beta,
            double threshold):
        self.R = R
        self.P = P
        self.Q = Q
        self.error = error
        self.steps = steps
        self.gamma = gamma
        self.beta = beta
        self.threshold = threshold

    cdef double get_rating_error(self, int user, int item):
        
        cdef:
            np.ndarray[DOUBLE_t, ndim=2, mode="c"] P = self.P
            np.ndarray[DOUBLE_t, ndim=2, mode="c"] Q = self.Q
        return self.R[user][item] - np.dot(P[:,user], Q[:,item])

    cdef double get_error(self, double beta):

        cdef:
            double error = 0.0
            np.ndarray[DOUBLE_t, ndim=2, mode="c"] P = self.P
            np.ndarray[DOUBLE_t, ndim=2, mode="c"] Q = self.Q

        for user_index in self.R:
            for item_index in self.R[user_index]:
                if self.R[user_index][item_index] == 0:
                    continue
                error += pow(self.get_rating_error(user_index, item_index), 2)

        error += beta * (np.linalg.norm(P) + np.linalg.norm(Q))
        return error

    def learning(self, int K, int steps = 30, double gamma = 0.005, double beta = 0.02, threshold = 0.1):

        cdef:
            double err = 0.0
            np.ndarray[DOUBLE_t, ndim=2, mode="c"] P = self.P
            np.ndarray[DOUBLE_t, ndim=2, mode="c"] Q = self.Q
            double error = 0.0

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

            error = self.get_error(beta)
            print error
            if error > threshold:
                self.nR = np.dot(np.transpose(P), Q) # 得られた評価値行列
                return
        
        self.nR = np.dot(np.transpose(P), Q)

    def predict(self, int user, int item):

        cdef:
            np.ndarray[DOUBLE_t, ndim=2, mode="c"] nR = self.nR
        return nR[user][item]
