import numpy as np


def MultipleLinearRegNormalForm(x, y):
    w = np.linalg.pinv(x.T @ x) @ x.T @ y
    return w


if __name__ == '__main__':
    dt = np.genfromtxt("D3.csv", delimiter=",")
    X = dt[:, 0:3]
    y = dt[:, 3]
    n_samples = len(X)
    ones = np.ones(n_samples)
    X = np.c_[ones, X]

    w = MultipleLinearRegNormalForm(X, y)
    print(w)
    print(np.dot(w, X[0]))
