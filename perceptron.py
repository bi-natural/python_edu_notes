# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

def net_input(X, w, b):
    return np.dot(X, w) + b

def activation(z):
    return np.where(z > 0, 1, -1)

def fit(X, y, 훈련횟수=10, 학습률=0.1):
    w = np.array([0.] * X.shape[1])
    b = 0.0
    
    error_history = []
    for i in range(훈련횟수):
        print('훈련횟수: {}'.format(i+1))
        sum_square_error = 0
        for xi, yi in zip(X, y):
            zi = net_input(xi, w, b)
            yi_pred = activation(zi)
            # 가중치 갱신
            error = yi - yi_pred
            sum_square_error += error **2
            update = 학습률 * error
            w += update * xi
            b += update
        print('w: {}\tb: {}\tError: {}'.format(
            w, b, sum_square_error))
        error_history.append(sum_square_error)
        
    return w, b, error_history

def main():
    iris = pd.read_csv('data/iris.data', header=None)
    data = iris[:100]
    shuffled_index = np.random.permutation(range(len(data)))
    train_index, test_index = shuffled_index[:70], shuffled_index[70:]
    X = data.loc[:, 0:3].values
    y = data[4]
    y = np.where(y == 'Iris-setosa', 1, -1)
    
    w, b, error_history = fit(X[train_index], y[train_index])
    
if __name__ == '__main__':
    main()
