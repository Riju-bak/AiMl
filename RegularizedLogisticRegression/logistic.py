import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self):
        self.iter_indices = []
        self.costs = []

    def init(self):
        self.costs = []
        self.iter_indices = []

    def fit(self, X, y, learning_rate=0.01, iters=100000, regularization_rate=1):
        self.init()
        n_samples = len(X)
        n_params = X.shape[1]   #compute n_params before appending a column of 1s to the X matrix

        ones = np.ones(n_samples)
        features = np.c_[ones, X]
        weights = np.zeros(features.shape[1]).T

        for i in range(iters):
            y_pred = self.y_predict(features, weights)
            error = y_pred - y
            dw = (2 / n_samples) * (features.T @ error)
            weights = (1 - learning_rate * regularization_rate / n_params) * weights - learning_rate * dw

            cost = self.compute_cost(n_samples, y, y_pred, n_params, regularization_rate, weights)
            self.costs.append(cost)
            self.iter_indices.append(i)

        self.plot_cost_vs_num_iterations()
        self.printLRMatrixFormPred(features, weights)

    def y_predict(self, features, weights):
        return self.sigmoid(features @ weights)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_cost(self, n_samples, y, y_pred, n_params, regularization_rate, weights):
        return (-1 / n_samples) * (y @ np.log(y_pred) + (1 - y) @ (np.log(1 - y_pred))) + \
            regularization_rate / (2 * n_params) * (weights @ weights)

    def plot_cost_vs_num_iterations(self):
        plt.plot(self.iter_indices, self.costs)
        plt.xlabel("#  of iterations")
        plt.ylabel("Cost")
        plt.show()

    def printLRMatrixFormPred(self, features, weights):
        def classify(feature, weights):
            probability = np.round(weights @ feature, 2)
            return 1 if probability >= 0 else 0

        for feature in features:
            # print(np.round(weights @ feature, 2), end=" ")
            print(classify(feature, weights), end=" ")
        print()
