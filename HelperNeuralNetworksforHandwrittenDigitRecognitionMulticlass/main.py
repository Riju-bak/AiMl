import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.python.keras.activations import linear, relu, sigmoid

plt.style.use('./deeplearning.mplstyle')

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from public_tests import *

from autils import *
from lab_utils_softmax import plt_softmax

np.set_printoptions(precision=2)

from autils import load_data


def my_softmax(z):
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """
    ### START CODE HERE ###
    exp_z = np.exp(z)
    s = sum(exp_z)
    a = exp_z / s

    ### END CODE HERE ###
    return a


def setup_model():
    tf.random.set_seed(1234)  # for consistent results

    model = Sequential(
        [
            Dense(25, activation="relu", name="L1"),
            Dense(15, activation="relu", name="L2"),
            Dense(10, activation="linear", name="L3"),
        ], name="my_model"
    )
    input_shape = (None, 400,)
    model.build(input_shape)
    return model


def make_prediction():
    image_of_two = X[1015]
    display_digit(image_of_two)

    prediction = model.predict(image_of_two.reshape(1, 400))  # prediction

    print(f" predicting a Two: \n{prediction}")
    print(f" Largest Prediction index: {np.argmax(prediction)}")

    prediction_p = tf.nn.softmax(prediction)

    print(f" predicting a Two. Probability vector: \n{prediction_p}")
    print(f"Total of predictions: {np.sum(prediction_p):0.3f}")

    yhat = np.argmax(prediction_p)

    print(f"np.argmax(prediction_p): {yhat}")


# Let's compare the predictions vs the labels for a random sample of 64 digits. This takes a moment to run.
def compare():
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # You do not need to modify anything in this cell

    m, n = X.shape

    fig, axes = plt.subplots(8, 8, figsize=(5, 5))
    fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]
    widgvis(fig)
    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20, 20)).T

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Predict using the Neural Network
        prediction = model.predict(X[random_index].reshape(1, 400))
        prediction_p = tf.nn.softmax(prediction)
        yhat = np.argmax(prediction_p)

        # Display the label above the image
        ax.set_title(f"{y[random_index, 0]},{yhat}", fontsize=10)
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=14)
    plt.show()

    print(f"{display_errors(model, X, y)} errors out of {len(X)} images")


if __name__ == '__main__':
    # load dataset
    X, y = load_data()
    print('The first element of X is: ', X[0])
    model = setup_model()
    model.summary()

    # BEGIN UNIT TEST
    test_model(model, 10, 400)
    # END UNIT TEST

    [layer1, layer2, layer3] = model.layers
    # Examine Weights shapes
    W1, b1 = layer1.get_weights()
    W2, b2 = layer2.get_weights()
    W3, b3 = layer3.get_weights()
    print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
    print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
    print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )

    history = model.fit(
        X, y,
        epochs=40
    )

    plot_loss_tf(history)

    make_prediction()

    compare()

# z = np.array([1., 2., 3., 4.])
# a = my_softmax(z)
# print(a)
