#!/usr/bin/env python3

import numpy as np


class Perceptron(object):
    """
    Perceptron model
    """

    def __init__(self):
        self.W = None
        self.b = None

    def train(self, X, y, learning_rate=1, verbose=False):
        X = np.asarray(X)
        N, D = X.shape
        y = np.asarray(y).reshape(N, 1)
        assert N == y.shape[0], 'labels not match!'
        # init
        self.W = np.zeros((D, 1), dtype=X.dtype)
        self.b = 0
        # train
        while True:
            idx = np.random.choice(N, 1)
            if y[idx] * (X[idx].dot(self.W) + self.b) <= 0:
                if verbose:
                    print("W: {}, b: {}".format(
                        np.squeeze(self.W), np.squeeze(self.b)), end=';  ')
                    print("Misclassification:", X[idx])
                self.W += learning_rate * y[idx] * X[idx].T
                self.b += learning_rate * y[idx]

            y_pred = np.sign(X @ self.W + self.b)
            if np.allclose(y_pred, y):
                if verbose:
                    print("W: {}, b: {}".format(
                        np.squeeze(self.W), np.squeeze(self.b)))
                break

    def predict(self, X):
        X = np.asarray(X)
        y_pred = np.sign(X @ self.W + self.b)
        return y_pred


class DualPerceptron(object):
    """
    Dual form of Perceptron
    """

    def __init__(self):
        self.alpha = None
        self.b = None

    def train(self, X, y, learning_rate=1., verbose=False):
        """
        train the classifier
        """
        X = np.asarray(X)
        N, D = X.shape
        y = np.asarray(y).reshape(N, 1)
        assert N == y.shape[0], 'labels not match!'
        # init
        self.alpha = np.zeros((N, 1), dtype=X.dtype)
        self.b = 0

        gram_matrix = X @ X.T

        while True:
            index = np.random.choice(N, 1)
            y_pred = alpha @ y * gram_matrix[:, index] + self.b
            if y[index] * y_pred <= 0:
                self.alpha[index] += learning_rate
                self.b += learning_rate * y[index]

    def predict(self, X):
        pass


if __name__ == '__main__':
    X = np.array([
        [3, 3],
        [4, 3],
        [1, 1]
    ], dtype=np.float32)
    y = np.array([1, 1, -1], dtype=np.float32)

    model = Perceptron()
    model.train(X, y, verbose=True)
