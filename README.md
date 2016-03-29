# Name

Collaborative Filtering

# Overview

協調フィルタリングを実装したプログラム

## Memory

ユーザーやアイテム間の類似度を基に行うメモリーベース協調フィルタリング
user_base ユーザー間の類似度を基に推薦を行う
item_base アイテム間の類似度を基に推薦を行う

## Model

Model_based CF: モデルを構築してから推薦を行うモデルベース協調フィルタリング

### Model/MatrixFactorization

Matrix Factorization Netflix Prizeで有名なMatrix Factorizationアルゴリズム

### Model/FactorizationMachine

ユーザーやアイテムの特徴量を入れることができるFactorization Machine

