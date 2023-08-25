# NOTE:  This is just a forward propagator, it assumes that optimal W and b matrices have been pre-computed
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dense(x, W, b):
    return sigmoid(x @ W + b)


def sequential(x):
    W = np.array([
        [1, -3, 5],
        [2, 4, -6]
    ])
    b = np.array([-1, 1, 2])

    W1 = W
    b1 = b

    W2 = np.array([
        [1, 4],
        [2, 5],
        [-3, -6]
    ])
    b2 = np.array([-1, 1])

    W3 = np.array([
        [1],
        [2]
    ])
    b3 = np.array([1])

    a1 = dense(x, W1, b1)
    a2 = dense(a1, W2, b2)
    a3 = dense(a2, W3, b3)

    return a3


if __name__ == '__main__':
    X = np.array([-2, 4])  # test-input
    print(sequential(X))
