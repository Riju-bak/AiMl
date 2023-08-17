import numpy as np
from linear import MultipleLinearRegression

if __name__ == '__main__':
    dt = np.genfromtxt("./data/D3.csv", delimiter=",")
    X = dt[:, 0:3]
    y = dt[:, 3]

    lr = MultipleLinearRegression()
    w = lr.fit(X, y)
    X = np.c_[np.ones(len(X)), X]
    print(f"{np.dot(w, X[0])}  {y[0]}\n")
