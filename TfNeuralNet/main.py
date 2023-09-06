from neural import NeuralNet
from utils import load_coffee_data

if __name__ == '__main__':
    X, y = load_coffee_data()
    print(X.shape)
    print(y.shape)
    nn = NeuralNet(X, y)
    nn.train()
    print(nn.predict(200, 13.9))
    print(nn.predict(200, 17))
