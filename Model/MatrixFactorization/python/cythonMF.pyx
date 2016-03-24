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
        
        cdef:
            np.ndarray[DOUBLE_t, ndim=2, mode="c"] R = self.R
            np.ndarray[DOUBLE_t, ndim=2, mode="c"] P = self.P
            np.ndarray[DOUBLE_t, ndim=2, mode="c"] Q = self.Q
        return R[user][item] - np.dot(P[:,user], Q[:,item])

    cdef double get_error(self, double beta):

        cdef:
            double error = 0.0
            long u_num = self.u_num
            long i_num = self.i_num
            long user_index
            long item_index
            np.ndarray[DOUBLE_t, ndim=2, mode="c"] R = self.R
            np.ndarray[DOUBLE_t, ndim=2, mode="c"] P = self.P
            np.ndarray[DOUBLE_t, ndim=2, mode="c"] Q = self.Q

        for user_index in xrange(u_num):
            for item_index in xrange(i_num):
                if R[user_index][item_index] == 0:
                    continue
                error += pow(self.get_rating_error(user_index, item_index), 2)

        error += beta * (np.linalg.norm(P) + np.linalg.norm(Q))
        return error

    def learning(self):

        cdef:
            double err = 0.0
            int K = self.K
            int steps = self.steps
            double gamma = self.gamma
            double beta = self.beta
            double threshold = self.threshold
            int step
            long u_num = self.u_num
            long i_num = self.i_num
            long user_index
            long item_index
            int k
            np.ndarray[DOUBLE_t, ndim=2, mode="c"] R = self.R
            np.ndarray[DOUBLE_t, ndim=2, mode="c"] P = self.P
            np.ndarray[DOUBLE_t, ndim=2, mode="c"] Q = self.Q
            double all_error = 0.0

        for step in xrange(steps):
            start = time.time()
            for user_index in xrange(u_num):
                for item_index in xrange(i_num):
                    if R[user_index][item_index] == 0:
                        continue
                    err = self.get_rating_error(user_index, item_index)
                    for k in xrange(K):
                        P[k][user_index] += gamma * (err*Q[k][item_index] - beta*P[k][user_index])
                        Q[k][item_index] += gamma * (err*P[k][user_index] - beta*Q[k][item_index])
            
            elapsed_time = time.time() - start
            print elapsed_time
            error = self.get_error(beta)
            print error
            if error < threshold:
                self.nR = np.dot(np.transpose(P), Q) # 得られた評価値行列
                return
        
        self.nR = np.dot(np.transpose(P), Q)

    def predict(self, int user, int item):

        cdef:
            np.ndarray[DOUBLE_t, ndim=2, mode="c"] nR = self.nR
        return nR[user][item]
