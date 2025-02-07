import numpy as np
import pandas as pd

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class _LogisticRegression():

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        if y.ndim == 2:
            y = y.reshape(-1)

        self.bias = 0

        i = 0

        while i < self.iterations:
            linear_predictions = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_predictions)

            dw = (1/m) * np.dot(X.T, (predictions - y))
            db = (1/m) * np.sum(predictions - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            i = i+1

    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        predictions = sigmoid(linear_predictions)

        predicted_values = [0 if y <= 0.5 else 1 for y in predictions]

        return predicted_values