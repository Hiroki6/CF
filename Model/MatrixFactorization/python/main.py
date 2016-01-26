# -*- coding:utf-8 -*-

import MF
import common
import math

"""
評価値行列を基にbasicMFクラスのオブジェクトを生成し、学習をして返す
@params(RateArray) 評価値行列
@return(basicMF) 学習されたbasicMF
"""
def create_basicMF(RateArray):
    
    basicMF = MF.basicMF(RateArray) # basicMFクラスのオブジェクト作成
    basicMF.learning(20, 500)

    return basicMF

def create_svd(RateArray):
    
    svd = MF.svd(RateArray)
    svd.learning(20)

    return svd

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
    RateArray = common.create_ratings() # 評価値行列作成
    learningData, testData = common.create_test_data(RateArray) # 教師データとテストデータ作成
    sum_MF_error = 0.0
    #sum_svd_error = 0.0
    basicMF = create_basicMF(learningData)
    #svd = create_svd(learningData)
    for user in testData:
        for item in testData[user]:
            sum_MF_error += pow((basicMF.predict(user, item) - testData[user][item]), 2)
            #sum_svd_error += pow((svd.predict(user, item) - testData[user][item]), 2)

    RMSE_MF = math.sqrt(sum_MF_error/300)
    #RMSE_svd = math.sqrt(sum_svd_error/300)
    print RMSE_MF
    #print RMSE_svd


