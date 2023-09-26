import autils
from autils import load_data
from neural import NeuralNet

if __name__ == '__main__':
    X, y = load_data()
    # print(X[0])
    # print(y)
    nn = NeuralNet()
    nn.train(X, y)

    prediction = nn.model.predict(X[0].reshape(1, 400))  # a zero
    print(f" predicting a zero: {prediction}")
    print(autils.predict(prediction))

    prediction = nn.model.predict(X[500].reshape(1, 400))  # a one
    print(f" predicting a one:  {prediction}")
    print(autils.predict(prediction))
