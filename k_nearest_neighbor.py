# Copyright (c) 2018 by pzhang. All Rights Reserved.

# chapter2 kNN of "Statistical Learning Method"

import numpy as np


class kNearestNeighbor(object):
    """
    a kNN classifier with L2 distance
    """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X:

        Returns:
        - y_pred:
        """
        dists = compute_distance(X)
        return self.predict_labels(dists, k=k)

    def compute_distance(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train

        Inputs:
        - X: A numpy array of shape (num_test, D)

        Returns:
        - dists:
        """
        num_train = self.X_train.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), dtype=X.dtype)
        for i in range(num_test):
            dists[i] = np.sum((X[i] - self.X_train)**2, axis=1)
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distance between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train)

        Returns:
        - y: A numpy array of shape (num_test, )
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            index = np.argsort(dists[i])[:k]
            closest_y = self.y_train[index]
            y_pred[i] = np.argmax(np.bincount(closest_y))


class KDTree(object):
    pass


if __name__ == "__main__":
    pass
