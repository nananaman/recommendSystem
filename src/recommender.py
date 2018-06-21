import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from src.BPRmodel import BPRmodel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path_data', help='Directory path where your data (ml-100k) are located.')
    parser.add_argument(
        'test_data', help='Name of your test data.')
    parser.add_argument('-H', '--trained_H_model', default=None, help='Path where your trained H model is located.')
    parser.add_argument('-W', '--trained_W_model', default=None, help='Path where your trained W model is located.')
    parser.add_argument('-o', '--path_out_dir', default='./',
                        help='Directory path where you want to create output folder.')
    parser.add_argument("-t", '--train_data', default=None,
                        help="Name of your train data.")
    args = parser.parse_args()

    # ml-100kのパスを格納
    #current = Path.home()
    path_data = Path(args.path_data)
    print('Path data: {}'.format(path_data.resolve()))
    # テストデータのパスを格納
    #testPath = current / Path('ml-100k/u1.test')
    testPath = path_data / Path(args.test_data)
    print('Path test data: {}'.format(testPath.resolve()))

    # itemのパスを格納
    itemPath = path_data / Path('u.item')
    print('Path item data: {}'.format(itemPath.resolve()))
    # userのパスを格納
    userPath = path_data / Path('u.user')
    print('Path user data: {}'.format(userPath.resolve()))

    # trainが指定されていればパスを格納
    if args.train_data:
        trainPath = path_data / Path('u1.base')
    # trainが指定されていなければ、モデルのパスを格納

    # 訓練済みHモデルが指定されていればパスを格納
    if args.trained_H_model:
        hPath = Path(args.trained_H_model)
        print('Path H model: {}'.format(hPath.resolve()))
    # trainが指定されてなく、かつ訓練済みHモデルが指定されていなければ、エラーを表示して中断
    elif not args.train_data:
        print('ERROR!: Please input path where your trained H model is located!')
        sys.exit()

    # 訓練済みWモデルが指定されていればパスを格納
    if args.trained_W_model:
        wPath = Path(args.trained_W_model)
        print('Path W model: {}'.format(wPath.resolve()))
    # trainが指定されてなく、かつ訓練済みWモデルが指定されていなければ、エラーを表示して中断
    elif not args.train_data:
        print('ERROR!: Please input path where your trained W model is located!')
        sys.exit()

    # 出力ディレクトリ作成
    out_dir = Path(args.path_out_dir) / Path('results')
    out_models_dir = out_dir / Path('models')
    out_models_dir.mkdir(parents=True, exist_ok=True)
    print('output_dir : {}'.format(out_dir.resolve()))

    # データ読み込み
    if args.train_data:
        train_data = trainPath.open()
    test_data = testPath.open()
    # itemのみデフォルトだとエラーが出るためencodingを指定
    item = itemPath.open(encoding="ISO-8859-1")
    user = userPath.open()

    # アイテム数、ユーザ数をカウント
    itemNum = sum(1 for line in item)
    userNum = sum(1 for line in user)
    print('itemNum : {}'.format(itemNum))
    print('userNum : {}'.format(userNum))

    # Dsを0で初期化
    train_Ds = np.zeros((userNum, itemNum))
    test_Ds = np.zeros((userNum, itemNum))

    # train用のDsを読み込み
    if args.train_data:
        for line in train_data:
            userID, itemID, rating, _ = map(int, line.split())
            # レートが登録されていればポジティブ
            train_Ds[userID - 1, itemID - 1] = 1

    # test用のDsを読み込み
    for line in test_data:
        userID, itemID, rating, _ = map(int, line.split())
        test_Ds[userID - 1, itemID - 1] = 1

    # モデルのインスタンス化
    model = BPRmodel(train_Ds, test_Ds, itemNum, userNum)

    # 訓練済みモデルが指定されていれば読み込み
    if args.trained_H_model and args.trained_W_model:
        model.load_model(hPath, wPath)

    # 訓練データが指定されていなければ
    if not args.train_data:
        # テストデータに対して推論
        predDs, loss = model.predict_test()
        preddir = out_dir / Path('predDs.npy')
        np.save(preddir, predDs)
        print('loss: {}'.format(loss))
    else:
        # 繰り返し回数を指定
        ITERATION_NUM = 50000
        # モデルをセーブする頻度を指定
        SAVE_INTERVAL = 5000
        # 進捗をディスプレイする頻度を指定
        DISPLAY_INTERVAL = 1000
        # プロット間隔を指定
        PLOT_INTERVAL = 10

        # ロスをプロットするためのx,y格納用リスト
        x = []
        y_train = []
        y_test = []

        # ログ保存用のファイル
        log_text = out_dir / Path('log.txt')
        # 作成
        log_text.write_text("Recommender's log file")

        # 指定iteration繰り返す
        for iteration in range(ITERATION_NUM):
            # モデルを更新する
            model.update()

            if iteration % PLOT_INTERVAL == 0:
                # プロット用のロス計算
                train_loss, test_loss = model.calc_loss()
                # ロスを記録
                with log_text.open(mode='a') as log:
                    log.write('--------------------------------\n')
                    log.write('iter : {}\n'.format(iteration))
                    log.write('train_loss : {}\n'.format(train_loss))
                    log.write('test_loss : {}\n'.format(test_loss))

                if iteration % DISPLAY_INTERVAL == 0:
                    # 画面に進捗を表示
                    print('iter : {}'.format(iteration))
                    print('train_loss : {}'.format(train_loss))
                    print('test_loss : {}'.format(test_loss))

                # モデルの保存
                if iteration % SAVE_INTERVAL == 0:
                    model.save_model(iteration, out_models_dir)

                # プロットするデータの保存
                x.append(iteration)
                y_train.append(train_loss)
                y_test.append(test_loss)

        # ロスをプロット
        plt.plot(x, y_train, marker='o', label='train')
        plt.plot(x, y_test, marker='x', label='test')
        plt.xlabel('iteration', fontsize=16)
        plt.ylabel('loss', fontsize=16)
        plt.savefig(str(out_dir / 'loss.png'))
        plt.show()


if __name__ == '__main__':
    main()
