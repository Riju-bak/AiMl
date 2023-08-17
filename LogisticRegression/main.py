import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression


def logisticRegressionUsingSkLearn(X, y):
    print("Using scikit-learn")
    lr_model = LogisticRegression()
    lr_model.fit(X, y)
    y_pred = lr_model.predict(X)
    print("Prediction on training set: ", y_pred)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def printLRMatrixFormPred(features, weights):
    def classify(feature, weights):
        probability = np.round(weights @ feature, 2)
        return 1 if probability >= 0 else 0

    for feature in features:
        # print(np.round(weights @ feature, 2), end=" ")
        print(classify(feature, weights), end=" ")
    print()


def plot_cost_vs_num_iterations(costs, iter_indices):
    plt.plot(iter_indices, costs)
    plt.xlabel("#  of iterations")
    plt.ylabel("Cost")
    plt.show()


def logisticRegressionMatForm(X, y, learning_rate=0.01, iters=100000):
    print("\nUsing my own matrix form code")
    n_samples = len(X)
    ones = np.ones(n_samples)
    features = np.c_[ones, X]
    weights = np.zeros(features.shape[1]).T

    costs = []
    iter_indices = []
    for i in range(iters):
        y_pred = YPredict(features, weights)
        error = y_pred - y
        dw = (2 / n_samples) * (features.T @ error)
        weights -= learning_rate * dw

        cost = ComputeCost(n_samples, y, y_pred)
        costs.append(cost)
        iter_indices.append(i)

    plot_cost_vs_num_iterations(costs, iter_indices)
    printLRMatrixFormPred(features, weights)


def ComputeCost(n_samples, y, y_pred):
    return (-1 / n_samples) * (y @ np.log(y_pred) + (1 - y) @ (np.log(1 - y_pred)))


def YPredict(features, weights):
    return sigmoid(features @ weights)


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

    logisticRegressionUsingSkLearn(X, y)
    logisticRegressionMatForm(X, y)
