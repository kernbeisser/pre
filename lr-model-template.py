import numpy as pd

class MyModel:
    #
    # init ?
    def __init__(self, X, y):
        self.w = None

    def fit(self, X, y, eta, T):
        self.w = np.random.randn(X.shape[1])

        for _ in range(T):
            y_hat = self.predict(X)
            self.w = self.w - eta*X.T.dot(y_hat -y)
    #

    def predict(self, X):
        return X.dot(self.w)
    #

    def cost(self, X, y):
        y_hat = self.predict(X)
        return (y_hat - y).dot(y_hat - y)

def main():
    url = ""
    X, y = load_data(url)
    model = MyModel()
    eta = 1;
    T = range(1, 100):
    model.fit(X, X, y, eta, T)

    if __name__ == '__main__':
        main()
