#!/usr/bin/env python3

import numpy as np


class Perceptron(object):
    """
    Perceptron model
    """

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, num_iters=100, verbose=False):
        N, dim = X.shape

    def predict(self, X):
        pass

    def loss(self, X, y):
        pass
