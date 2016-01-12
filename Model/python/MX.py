# -*- coding:utf-8 -*-

import numpy


def get_rating_error(r, p, q):
    return r - numpy.dot(p, q)

"""
目的関数
@param(R) 評価値行列
@param(P) ユーザー潜在ベクトル
@param(Q) アイテム潜在ベクトル
@param(beta) 正規化係数
@return(error) 誤差
"""
def get_error(R, P, Q, beta):
    error = 0.0
    for i in range(len(R)):
        for j in xrange(len(R[i])):
            if R[i][j] == 0:
                continue
            error += pow(get_rating_error(R[i][j], P[:,i], Q[:,j]), 2)
    # 正規化項
    error += beta * (numpy.linalg.norm(P) + numpy.linalg.norm(Q))
    return error

"""
ユーザー行列Pとアイテム行列Qを最適化によって求める
@param(R) 評価値行列
@param(K) 疑似行列のベクトル
@param(steps) 学習回数
@param(alpha) 学習係数
@param(beta) 正規化係数
@param(threshold) 誤差の閾値
@return(P,Q) 最適化によって求められたユーザー行列Pとアイテム行列Q
"""
def matrix_factorizaion(R, K, steps = 5000, gamma = 0.0002, beta = 0.02, threshold = 0.001):
    P = numpy.random.rand(K, len(R))
    Q = numpy.random.rand(K, len(R[0]))
    for step in xrange(steps):
        for i in xrange(len(R)):
            if R[i][j] == 0:
                continue
            err = get_rating_error(R[i][j], P[:,i], Q[:,j])
            for k in xrange(K):
                P[k][i] += gamma * (err*Q[k][j] - beta*P[k][i])
                Q[k][j] += gamma * (err*P[k][i] - beta*Q[k][j])
        error = get_error(R, P, Q, beta)
        if error < threshold:
            break
    return P, Q
