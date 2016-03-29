# Name

Collaborative Filtering

# Overview

協調フィルタリングを実装したプログラム

## /Memory

ユーザーやアイテム間の類似度を基に行うメモリーベース協調フィルタリング
user_base ユーザー間の類似度を基に推薦を行う

<img src="https://github.com/Hiroki6/Collaborative-Filtering/blob/master/images/userbase.png" width="300">

item_base アイテム間の類似度を基に推薦を行う

<img src="https://github.com/Hiroki6/Collaborative-Filtering/blob/master/images/itembase.png" width="300">

## /Model

Model_based CF: モデルを構築してから推薦を行うモデルベース協調フィルタリング

### /Model/MatrixFactorization

Matrix Factorization Netflix Prizeで有名なMatrix Factorizationアルゴリズム

<img src="https://github.com/Hiroki6/Collaborative-Filtering/blob/master/images/MF_kai.png" width="600">

### /Model/FactorizationMachine

ユーザーやアイテムの特徴量を入れることができるFactorization Machine

