from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import numpy as np
import os
import time
from beautifultable import BeautifulTable


def load_length_data(opt):
    data = pd.read_csv('length_'+opt+'.csv')
    df = pd.DataFrame(data,
                      columns=[
                          'dist_centroids_x',
                          'dist_centroids_y',
                          'length_ratio',
                          'disk_length',
                          'disk_center',
                          'pig_center',
                          'pig_hull_center',
                          'pig_area',
                          # 'pig_perimeter',
                          # 'pig_short_axis',
                          'pig_thickness',
                          'angle',
                          'disk_is_inside',
                          'real_pig_length'
                      ],
                      index=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx, ...]
    y = y[idx]

    return X, y, df


def load_weight_data(opt):
    data = pd.read_csv('weight_'+opt+'.csv')
    df = pd.DataFrame(data,
                      columns=[
                          'dist_centroids_x',
                          'dist_centroids_y',
                          'length_ratio',
                          'disk_length',
                          # 'disk_center',
                          # 'pig_center',
                          'pig_hull_center',
                          'pig_area',
                          'pig_perimeter',
                          # 'pig_short_axis',
                          'pig_thickness',
                          'angle',
                          'pig_color_gray',
                          # 'pig_color_br',
                          # 'area_ratio',
                          'disk_is_inside',
                          'real_pig_weight'
                      ],
                      index=None)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx, ...]
    y = y[idx]

    return X, y, df


def main():

    eval_flag = False
    write_flag = False
    seed = 5

    print('========================= Length model =========================')
    X_train, y_train, df = load_length_data('train')

    if eval_flag:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print('X_train shape', X_train.shape)

    forest = RandomForestRegressor(
            n_estimators=1000,
            criterion='mse',
            max_depth=15,
            min_samples_leaf=3,
            max_features=0.4,
            random_state=1,
            n_jobs=-1
        )
    forest.fit(X_train, y_train)

    importance = forest.feature_importances_
    table = BeautifulTable()
    table.column_headers = ["feature", "importance"]
    print('RF feature importance:')
    for i, cols in enumerate(df.iloc[:, :-1]):
        table.append_row([cols, round(importance[i], 3)])
    print(table)

    if eval_flag:
        pred = forest.predict(X_val)
        M = np.mean(abs(pred - y_val) / y_val)
        print('RF val mean error: {}'.format(M))

    X_test, y_test, _ = load_length_data('test')
    pred = forest.predict(X_test)
    M = np.mean(abs(pred - y_test) / y_test)
    print('X_test shape', X_test.shape)
    print('RF test mean error: {}'.format(M))

    if write_flag:
        with open("RF_length.pkl", "wb") as f:
            pickle.dump(forest, f)
            print('Model saved.')

    print('========================= Weight model =========================')
    X_train, y_train, df = load_weight_data('train')
    if eval:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
    X_test, y_test, _ = load_weight_data('test')

    print('X_train shape', X_train.shape)
    forest = RandomForestRegressor(
            n_estimators=1000,
            criterion='mse',
            max_depth=15,
            min_samples_leaf=3,
            max_features=0.4,
            random_state=1,
            n_jobs=-1
        )
    forest.fit(X_train, y_train)

    importance = forest.feature_importances_
    table = BeautifulTable()
    table.column_headers = ["feature", "importance"]
    print('RF feature importance:')
    for i, cols in enumerate(df.iloc[:, :-1]):
        table.append_row([cols, round(importance[i], 3)])
    print(table)

    if eval_flag:
        pred = forest.predict(X_val)
        M = np.mean(abs(pred - y_val) / y_val)
        print('RF val mean error: {}'.format(M))

    pred = forest.predict(X_test)
    M = np.mean(abs(pred - y_test) / y_test)
    print('X_test shape', X_test.shape)
    print('RF test mean error: {}'.format(M))

    if write_flag:
        with open("RF_weight.pkl", "wb") as f:
            pickle.dump(forest, f)
            print('Model saved.')


if __name__ == '__main__':
    main()
