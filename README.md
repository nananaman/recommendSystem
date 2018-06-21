# Bayesian Personalized Ranking

## 概要
MovieLens 100K DatasetをBayesian Personalized Rankingで学習するコード。

## 説明
映画のレコメンドシステムを実現するためのアルゴリズムとしてBayesian Personalized Ranking (以下BPR) を実装した。
データセットにはMovieLens 100K Datasetを用いた。
このデータセットでは各ユーザーが各映画について5段階評価のレートをつけているが、本実装ではこれを無視し、評価をしていれば1、していなければ0としている。
これは、BPRが各アイテムについて各ユーザーがPositiveであるか否かという2値データを扱うものだからである。

## インストール
```
pip install ./recommendSystem
```

## 使い方

### 予測
```
recommender {データセットのディレクトリ} {テストデータの名前} -H {Hのモデルのパス} -W {Wのモデルのパス}
```

### 学習
```
recommender {データセットのディレクトリ} {テストデータの名前} -t {訓練データの名前}
```

## 使用例

### 予測
```
recommender '~/ml-100k' 'u1.test' -H '~/results/models/H_45000_iter.npy' -W '~/results/models/W_45000_iter.npy'
```

### 学習
```
recommender '~/ml-100k' 'u1.test' -t 'u1.base'
```
