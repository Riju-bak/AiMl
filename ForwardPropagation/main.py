# NOTE:  This is just a forward propagator, it assumes that optimal W and b matrices have been pre-computed
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(x))


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

    W2 = W
    b2 = b

    W3 = W
    b3 = b

    W4 = W
    b4 = b

    a1 = dense(x, W1, b1)
    a2 = dense(a1, W2, b2)
    a3 = dense(a2, W3, b3)
    a4 = dense(a3, W4, b4)

    return a4


if __name__ == '__main__':
    X = np.array([-2, 4])  # test-input
    print(sequential(X))
