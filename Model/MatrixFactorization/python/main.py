# -*- coding:utf-8 -*-

import MF
import common

"""
評価値行列を基にbasicMFクラスのオブジェクトを生成し、学習をして返す
@params(RateArray) 評価値行列
@return(basicMF) 学習されたbasicMF
"""
def create_basicMF(RateArray):
    
    basicMF = MF.basicMF(RateArray) # basicMFクラスのオブジェクト作成
    basicMF.learning(1000)
    print(basicMF.nR)

    return basicMF

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
    basicMF = create_basicMF(RateArray)

