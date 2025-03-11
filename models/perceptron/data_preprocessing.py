# data_preprocessing.py

import numpy as np
from sklearn.datasets import fetch_openml

def load_mnist():
    """
    Loads the MNIST dataset using fetch_openml, default parser => requires pandas installed.
    Returns:
        X: Array of shape (n_samples, 784) with pixel features.
        y: Array of shape (n_samples,) with digit labels.
    """
    # This will use pandas internally, so make sure you have `pip install pandas`.
    mnist = fetch_openml('mnist_784', version=1)  # default => requires pandas
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int32)
    return X, y

def preprocess_data(X):
    """
    Normalizes pixel values to [0, 1] and adds a bias term.
    Each image (28x28) is flattened into a 785D vector, with first element = 1 (bias).
    """
    X = X / 255.0
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    return X_bias
