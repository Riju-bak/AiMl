import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def MultipleLinearRegNormalForm(x, y):
    # n_samples = len(x)
    # ones = np.ones(n_samples)
    # x = np.c_[ones, x]
    w = np.linalg.inv(x.T @ x) @ x.T @ y
    return w


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


def MultipleLinearRegMatForm(x, y, learning_rate=0.0001, iters=100000):
    # x_mean = np.mean(x, axis=0)
    # x_std = np.std(x, axis=0)
    # x_normalized = (x - x_mean) / x_std
    # x = x_normalized

    n_samples = len(x)
    # ones = np.ones(n_samples)
    # features = np.c_[ones, x]
    features = x
    weights = np.zeros(features.shape[1]).T

    iters_ind = []
    cost_per_iteration = []
    for i in range(iters):
        y_predicted = YPredict(features, weights)
        error = y_predicted - y

        cost = ComputeCost(error, n_samples)
        cost_per_iteration.append(cost)
        iters_ind.append(i)

        dw = (2 / n_samples) * (features.T @ error)
        weights -= learning_rate * dw

    plt.plot(iters_ind, cost_per_iteration)
    plt.xlabel("# of iterations")
    plt.ylabel("Cost")
    plt.show()
    return weights.T


def ComputeCost(error, n_samples):
    return (2 / n_samples) * (error @ error)


def YPredict(features, weights):
    return features @ weights


def MLRSkLearn(x, y):
    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(X, y)

    # Get the parameter vector (coefficients)
    parameter_vector = np.append(model.intercept_, model.coef_)
    return parameter_vector


if __name__ == '__main__':
    dt = np.genfromtxt("D3.csv", delimiter=",")
    X = dt[:, 0:3]
    y = dt[:, 3]

    print(f"Using gradient-descent (my own code)")
    w, b = MultipleLinearReg(X, y)
    print(f"{np.dot(w, X[0]) + b}  {y[0]}\n")

    print(f"Using sklearn")
    w = MLRSkLearn(X, y)

    n_samples = len(X)
    ones = np.ones(n_samples)
    X = np.c_[ones, X]

    print(f"{np.dot(w, X[0])}  {y[0]}\n")

    # w, b = MultipleLinearReg(X, y)
    # print(f"{w}\n")
    # print(np.dot(w, X[1]) + b)

    print("Using normal form")
    w = MultipleLinearRegNormalForm(X, y)
    print(f"{np.dot(w, X[0])}  {y[0]}\n")

    print("Using gradient descent matrix form")
    w = MultipleLinearRegMatForm(X, y)
    print(f"{np.dot(w, X[0])}  {y[0]}\n")
