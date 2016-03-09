# -*- coding:utf-8 -*-

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DOUBLE_t
ctypedef np.int_t INT_t

cdef class basicMF:

    cdef np.ndarray P
    cdef np.ndarray Q
    def __cinit__(self, R):
        self.R = R

    cpdef get_rating_error(self, int user, int item):
        return self.R[user][item] - np.dot(self.R[:,user], self.Q[:,item])

    cpdef get_error(self, double beta):
        double self.error = 0.0

        for user_index in self.R:
            for item_index in self.R[user_index]:
                if self.R[user_index][item_index] == 0:
                    continue
                self.error += pow(self.get_rating_error(user_index, item_index), 2)

        self.error += beta * (np.linalg.norm(self.P) + np.linalg.norm(self.Q))

    cpdef learning(self, int K, int steps = 30, double gamma = 0.005, double beta = 0.02, threshold = 0.1):
        self.P = np.random.rand(K, len(self.R))
        self.Q = np.random.rand(K, len(self.R[0]))
        
        double err = 0.0
        for step in xrange(steps):
            for user_index in self.R:
                for item_index in self.R[user_index]:
                    if self.R[user_index][item_index] == 0:
                        continue
                    err = self.get_rating_error(user_index, item_index)
                    for k in xrange(K):
                        self.P[k][user_index] += gamma * (err*self.Q[k][item_index] - beta*self.P[k][user_index])
                        self.Q[k][item_index] += gamma * (err*self.P[k][user_index] - beta*self.Q[k][item_index])

            self.get_error(beta)
            print self.error
            if self.error > threshold:
                self.nR = np.dot(self.P.T, self.Q) # 得られた評価値行列
                return 

        self.nR = np.dot(self.P.T, self.Q)

    cpdef predict(self, int user, int item):
        return self.nR[user][item]
