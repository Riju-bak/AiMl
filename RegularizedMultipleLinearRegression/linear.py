import numpy as np
import matplotlib.pyplot as plt


class MultipleLinearRegression:
    def __init__(self):
        self.costs = []
        self.iter_indices = []

    def fit(self, x, y, learning_rate=0.01, iters=1000, regularization_rate=1):
        n_samples = x.shape[0]
        n_params = x.shape[1]

        ones = np.ones(n_samples)
        features = np.c_[ones, x]
        weights = np.zeros(features.shape[1]).T

        for i in range(iters):
            y_pred = self.y_predict(features, weights)
            error = y_pred - y
            dw = (2 / n_samples) * (features.T @ error)
            weights = (1 - learning_rate * regularization_rate / n_params) * weights - learning_rate * dw

            cost = self.compute_cost(error, n_samples, n_params, regularization_rate, weights)
            self.costs.append(cost)
            self.iter_indices.append(i)

        self.plot_cost_vs_num_iterations()
        return weights

    def compute_cost(self, error, n_samples, n_params, regularization_rate, weights):
        return (2 / n_samples) * (error @ error) + regularization_rate / (2 * n_params) * (weights @ weights)

    def y_predict(self, features, weights):
        return features @ weights

    def plot_cost_vs_num_iterations(self):
        plt.plot(self.iter_indices, self.costs)
        plt.xlabel("# of iterations")
        plt.ylabel("Cost")
        plt.show()
