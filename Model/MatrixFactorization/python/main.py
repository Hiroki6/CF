# -*- coding:utf-8 -*-
import sys
sys.path.append("cython-mf")
sys.path.append("mf")
sys.dont_write_bytecode = True 
import mf
import cylibmf
import common
import math


"""
評価値行列を基にbasicMFクラスのオブジェクトを生成し、学習をして返す
@params(rate_matrix) 評価値行列
@return(basicMF) 学習されたbasicMF
"""
def create_basic_mf(rate_matrix):
    
    print "学習開始"
    basicMF = mf.BasicMF(rate_matrix) # basicMFクラスのオブジェクト作成
    basicMF.learning(20, 500)

    return basicMF

def create_cy_mf(rate_matrix):

    print "学習開始"
    cyMF = cylibmf.BasicMF(rate_matrix)
    cyMF.learning(20, 500)

    return cyMF

def create_svd(rate_matrix):
    
    svd = mf.Svd(rate_matrix)
    svd.learning(20)

    return svd

def calc_rmse(learnedObj, testData):

    print "精度計測開始"
    sum_error = 0.0
    for test in testData:
        sum_MF_error += pow((learnedObj.predict(test[0], test[2]) - test[3]), 2)

    rmse = math.sqrt(sum_error/len(testData))
    print rmse 

"""
userに対するアイテムへの予測評価値をランキングで表示する
@params(basicMF) 学習済みのbasicMFオブジェクト
@params(user) 予測したいユーザー
@return(rankings) アイテムと評価値のタプル
"""
def predict_basic_mf(basicMF, user):
    
    rankings = basicMF.recommends(user)
    print(rankings)
    return rankings

if __name__ == "__main__":
    print "データ作成"
    rate_matrix, usermap, itemmap = common.create_matrix() # 評価値行列作成
    learningData, testData = common.create_test_data(rate_matrix) # 教師データとテストデータ作成
    cyMF = create_cy_mf(learningData)
    calc_rmse(cyMF, testData)


