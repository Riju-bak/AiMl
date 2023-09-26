import numpy as np

from neural import NeuralNet
from utils import load_coffee_data

if __name__ == '__main__':
    X, y = load_coffee_data()
    print(X)
    print(y)
    nn = NeuralNet()
    nn.train(X, y)
    x_test = np.array(
        [
            [185, 12.6],  # good roast
            [200, 17]  # bad roast
        ]
    )
    y_test = np.array(
        [
            [1],
            [0]
        ]
    )

    nn.evaluate(x_test, y_test)
    print(nn.predict(x_test))
