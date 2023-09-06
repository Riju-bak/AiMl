import tensorflow as tf
import numpy as np


# TODO: Fix prediction
class NeuralNet:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.W = []

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=2, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid'),
        ])

    def train(self):
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy())
        self.model.fit(self.X, self.y, epochs=100)

    def predict(self, temp, time):
        return self.model.predict(
            np.array(
                [
                    [200, 13.9],  # postive example
                    [200, 17]
                ]
            )
        )
