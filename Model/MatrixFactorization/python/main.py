# -*- coding:utf-8 -*-

import MF
import common
import math

"""
評価値行列を基にbasicMFクラスのオブジェクトを生成し、学習をして返す
@params(rate_matrix) 評価値行列
@return(basicMF) 学習されたbasicMF
"""
def create_basicMF(rate_matrix):
    
    print "学習開始"
    basicMF = MF.basicMF(rate_matrix) # basicMFクラスのオブジェクト作成
    basicMF.learning(20, 500)

    return basicMF

def create_cyMF(rate_matrix):

    print "学習開始"
    cyMF = MF.Cy_basicMF(rate_matrix)
    cyMF.learning(20, 500)

    return cyMF

def create_svd(rate_matrix):
    
    svd = MF.svd(rate_matrix)
    svd.learning(20)

    return svd

def calc_RMSE(learnedObj, testData):

    print "精度計測開始"
    sum_error = 0.0
    for test in testData:
        sum_MF_error += pow((learnedObj.predict(test[0], test[2]) - test[3]), 2)

    RMSE = math.sqrt(sum_error/len(testData))
    print RMSE

"""
userに対するアイテムへの予測評価値をランキングで表示する
@params(basicMF) 学習済みのbasicMFオブジェクト
@params(user) 予測したいユーザー
@return(rankings) アイテムと評価値のタプル
"""
def predict_basicMF(basicMF, user):
    
    rankings = basicMF.recommends(user)
    print(rankings)
    return rankings

if __name__ == "__main__":
    print "データ作成"
    rate_matrix, usermap, itemmap = common.create_matrix() # 評価値行列作成
    learningData, testData = common.create_test_data(rate_matrix) # 教師データとテストデータ作成
    basicMF = create_basicMF(learningData)
    calc_RMSE(basicMF, testData)
    cyMF = create_cyMF(learningData)
    calc_RMSE(cyMF, testData)


