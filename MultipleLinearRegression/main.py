import numpy as np
import pandas as pd


# This is correct
def MultipleLinearRegNormalForm(x, y):
    # n_samples = len(x)
    # ones = np.ones(n_samples)
    # x = np.c_[ones, x]
    w = np.linalg.inv(x.T @ x) @ x.T @ y
    return w


# Probably wrong
def MultipleLinearReg(x, y, learning_rate=0.001, iters=10000):
    # Normalize input features
    # x_mean = np.mean(x, axis=0)
    # x_std = np.std(x, axis=0)
    # x_normalized = (x - x_mean) / x_std
    # x = x_normalized

    features = x

    # Initial value of w and b
    n_params = features.shape[1]
    w = np.zeros(n_params)
    b = 0
    n_samples = features.shape[0]

    def ddw(j):
        sig = 0
        for i in range(n_samples):
            sig += (np.dot(w, features[i]) + b - y[i]) * features[i, j]
        return sig

    def ddb():
        sig = 0
        for i in range(n_samples):
            sig += (np.dot(w, features[i]) + b) - y[i]
        return sig

    for i in range(iters):
        tmp_w = w
        for j in range(n_params):
            tmp_w[j] -= learning_rate / n_samples * ddw(j)
        tmp_b = b - learning_rate / n_samples * ddb()
        w = tmp_w
        b = tmp_b

        # Debugging prints
        # if i % 10 == 0:  # Print every 10 iterations
        #     print(f"Iteration {i}: w = {w}, b = {b}")
    return w, b


# THIS IS CORRECT
def MultipleLinearRegMatForm(x, y, learning_rate=0.001, iters=1000):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_normalized = (x - x_mean) / x_std
    x = x_normalized

    n_samples = len(x)
    # ones = np.ones(n_samples)
    # features = np.c_[ones, x]
    features = x
    weights = np.zeros(features.shape[1]).T

    for i in range(iters):
        y_predicted = features @ weights
        error = y_predicted - y
        dw = (2 / n_samples) * (features.T @ error)
        weights -= learning_rate * dw
    return weights.T


if __name__ == '__main__':
    file_path = 'kc_house_data.csv'
    data_frame = pd.read_csv(file_path)
    y = np.array(data_frame['price'][:-10])

    columns_to_skip = ['id', 'date', 'price']
    data_frame = pd.read_csv(file_path, usecols=lambda col: col not in columns_to_skip)
    X = np.array(data_frame[:-10])

    # w, b = MultipleLinearReg(X, y)
    # print(f"{w}\n")
    # print(np.dot(w, X[1]) + b)

    ones = np.ones(len(X))
    X = np.c_[ones, X]
    w = MultipleLinearRegNormalForm(X, y.T)
    print(w)
    print((w @ X[0]))
