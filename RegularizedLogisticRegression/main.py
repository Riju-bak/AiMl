import numpy as np

from logistic import LogisticRegression

if __name__ == '__main__':
    X = np.array(
        [
            [0.5, 0.5],
            [1, 1],
            [1.5, 0.5],
            [3, 0.5],
            [2, 2],
            [1, 2.5]
        ])

    y = np.array([0, 0, 0, 1, 1, 1])
    lr = LogisticRegression()
    lr.fit(X, y)
