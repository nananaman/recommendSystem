# Bayesian Personalized Ranking

## 概要
MovieLens 100K DatasetをBayesian Personalized Rankingで学習するコード。

## 説明
映画のレコメンドシステムを実現するためのアルゴリズムとしてBayesian Personalized Ranking (以下BPR) を実装した。
データセットにはMovieLens 100K Datasetを用いた。
このデータセットでは各ユーザーが各映画について5段階評価のレートをつけているが、本実装ではこれを無視し、評価をしていれば1、していなければ0としている。
これは、BPRで扱うデータが各アイテムについて各ユーザーがPositiveであるか否かという2値のものだからである。

## インストール
```
pip install ./recommendSystem
```

## 使い方

### 予測
```
recommender {データセットのディレクトリ} {テストデータのファイル名} -H {Hのモデルのパス} -W {Wのモデルのパス}
```

### 学習
```
recommender {データセットのディレクトリ} {テストデータのファイル名} -t {訓練データのファイル名}
```

## 使用例

### 予測
```
recommender '~/ml-100k' 'u1.test' -H '~/recommendSystem/sample/H_45000_iter.npy' -W '~/recommendSystem/sample/W_45000_iter.npy'
```

### 学習
```
recommender '~/ml-100k' 'u1.test' -t 'u1.base'
```
