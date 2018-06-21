import random
import numpy as np
from pathlib import Path


def sigmoid(x):
    '''
    sigmoid関数
    :param x : x
    :return : sigmoid(x)
    '''
    # クリッピングのレンジを設定
    sigmoid_range = 34.5387763910684
    # NaNを出力しないようにクリッピング
    if x <= -sigmoid_range:
        return 1e-15
    if x >= sigmoid_range:
        return 1. - 1e-15
    return 1. / (1. + np.exp(-x))


class BPRmodel:
    def __init__(self, train_Ds, test_Ds, itemNum, userNum, k=20, alpha=0.1, lam=0.1):
        '''
        modelのインスタンス化
        :param train_Ds : 訓練用のDs
        :param test_Ds : テスト用のDs
        :param itemNum : アイテム数
        :param userNum : ユーザー数
        :param k : ファクター数
        :param alpha : 学習率
        :param lam : 正則化のハイパーパラメータlambda
        '''
        self.Ds = train_Ds
        self.testDs = test_Ds
        self.itemNum = itemNum
        self.userNum = userNum
        # ハイパーパラメータ
        self.k = k
        self.alpha = alpha
        self.lam = lam
        # パラメータの初期化
        self.W = np.random.rand(userNum, self.k)
        self.H = np.random.rand(itemNum, self.k)

    def update(self):
        '''
        modelのアップデート
        '''
        # ランダムなu, i, jを用意
        u, i, j = self.choice()
        # 各ファクターを更新
        for f in range(self.k):
            # 勾配計算
            grad = self.H[i, f] - self.H[j, f]
            # x_uiの予測値を計算
            x_ui = self.predict(u, i)
            # x_ujの予測値を計算
            x_uj = self.predict(u, j)
            # x_uijの予測値を計算
            x_uij = x_ui - x_uj
            # W_ufを更新
            self.W[u, f] -= self.alpha * \
                (sigmoid(x_uij) * grad + self.lam * self.W[u, f])

            # 勾配計算
            grad = self.W[u, f]
            # H_ifを更新
            self.H[i, f] -= self.alpha * \
                (sigmoid(x_uij) * grad + self.lam * self.H[i, f])
            # H_jfを更新
            self.H[j, f] -= self.alpha * \
                (sigmoid(x_uij) * -grad + self.lam * self.H[j, f])

    def choice(self):
        '''
        ランダムなu, i, jを用意
        :return : ランダムなu, i, j
        '''
        # Iu^+を抽出する準備
        Iu_plus = []
        # Iu_plusが要素を持つまでuを選びなおす
        while len(Iu_plus) == 0:
            # ランダムなuを用意
            u = np.random.randint(self.userNum)
            # Iuを抽出
            Iu = self.Ds[u]
            # Iu内の各アイテムについて
            for k in range(len(Iu)):
                # それが1であればIu^+に入れる
                if Iu[k] == 1:
                    Iu_plus.append(k)
        # Iu_minusを作成
        Iu_minus = list(set(range(len(Iu))) - set(Iu_plus))
        # Iu_plus内のランダムなアイテムiを選択
        i = random.choice(Iu_plus)
        # Iu_minus内のランダムなアイテムjを選択
        j = random.choice(Iu_minus)
        return u, i, j

    def predict(self, u, i):
        '''
        x_uiを計算する
        :param u : ユーザー
        :param i : アイテム
        :return : x_ui
        '''
        return self.W[u].dot(self.H[i])

    def add_loss(self, Ds, u, loss, n):
        Iu = Ds[u]
        # Iu^+を抽出
        Iu_pluss = []
        for k in range(len(Iu)):
            if Iu[k] == 1:
                Iu_pluss.append(k)
        # Iu^+がなければスキップ
        if len(Iu_pluss) != 0:
            for i in Iu_pluss:
                # x_uiの予測
                x_ui = self.predict(u, i)
                # 2乗誤差を取る
                loss += (Ds[u, i] - x_ui) ** 2
                # カウンターを増やす
                n += 1
        return loss, n

    def calc_loss(self):
        '''
        訓練データとテストデータについてロス (RMSE) の計算を行う
        :return : 訓練データのロスの平均、テストデータのロスの平均
        '''
        # ロスの初期化
        train_loss = 0
        test_loss = 0
        # ロスの平均を取るためにカウンターの準備
        n_train = 0
        n_test = 0
        # ユーザーごとに反復
        for u in range(self.userNum):
            # train
            # ロスを計算してカウンターを増やす
            train_loss, n_train = self.add_loss(self.Ds, u, train_loss, n_train)

            # test
            test_loss, n_test = self.add_loss(self.testDs, u, test_loss, n_test)

        # RMSEを取る
        train_loss = np.sqrt(train_loss / n_train)
        test_loss = np.sqrt(test_loss / n_test)
        return train_loss, test_loss

    def predict_test(self):
        '''
        テストデータについてDsの予測とロスを返す
        :return : Dsの予測値、ロスの平均
        '''
        test_loss = 0
        n_test = 0
        # Dsの予測値の初期化
        predDs = np.zeros_like(self.testDs)
        for u in range(self.userNum):
            # Iuを抽出
            Iu = self.testDs[u]
            # Iu^+を抽出
            Iu_plus = []
            for k in range(len(Iu)):
                if Iu[k] == 1:
                    Iu_plus.append(k)
            # Iu^+がなければスキップ
            if len(Iu_plus) != 0:
                for i in Iu_plus:
                    x_ui = self.predict(u, i)
                    # predDsに予測結果を記録
                    predDs[u, i] = x_ui
                    test_loss += (self.testDs[u, i] - x_ui) ** 2
                    n_test += 1
        test_loss = np.sqrt(test_loss / n_test)
        return predDs, test_loss

    def save_model(self, iteration, path):
        '''
        modelをsaveする
        :param iteration : 何iteration目なのか
        :param path : 出力先
        '''
        wdir = path / Path('W_{}_iter.npy'.format(iteration))
        hdir = path / Path('H_{}_iter.npy'.format(iteration))
        np.save(wdir, self.W)
        np.save(hdir, self.H)

    def load_model(self, hPath, wPath):
        '''
        modelをloadする
        :param hPath : Hのパス
        :param wPath : Wのパス
        '''
        self.H = np.load(hPath)
        self.W = np.load(wPath)
