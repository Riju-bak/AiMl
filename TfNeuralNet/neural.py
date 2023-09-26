import tensorflow as tf
import numpy as np


# TODO: Fix prediction
class NeuralNet:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=25, activation='sigmoid'),
            tf.keras.layers.Dense(units=10, activation='sigmoid'),
            tf.keras.layers.Dense(units=5, activation='sigmoid'),
            tf.keras.layers.Dense(units=1, activation='sigmoid'),
        ])

    def train(self, X, y):
        X = self.normalize_data(X)
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy())
        self.model.fit(X, y, epochs=100)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test):
        print(self.model.evaluate(x_test))

    def normalize_data(self, X):
        norm_l = tf.keras.layers.Normalization(axis=-1)
        norm_l.adapt(X)  # learns mean, variance
        Xn = norm_l(X)
        return Xn
