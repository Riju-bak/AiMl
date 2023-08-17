# Simple Linear Regression
import numpy as np
import matplotlib.pyplot as plt


def TwoDLineFunc(x, w, b):
    return w * x + b


def LinearReg(x, y):
    w, b = 0, 0  # initial value of w and b
    a = 0.001  # The learning rate
    for i in range(10000):
        tmp_w = w - a * np.sum((w * x + b - y) * x)
        tmp_b = b - a * np.sum(w * x + b - y)
        w = tmp_w
        b = tmp_b
    return w, b


if __name__ == '__main__':
    x = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
    y = np.array([40, 43, 55, 60, 68, 77, 82, 86, 94])

    plt.scatter(x, y)

    w, b = LinearReg(x, y)
    plt.plot(x, TwoDLineFunc(x, w, b), color='red')
    plt.show()
