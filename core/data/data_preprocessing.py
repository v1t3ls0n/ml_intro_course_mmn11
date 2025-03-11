# data_preprocessing.py
import numpy as np
from sklearn.model_selection import train_test_split
from core.data.mnist_loader import load_mnist

def preprocess_data(X, add_bias=True, normalize=True):
    """
    Preprocesses MNIST data for model training.

    Args:
        X (ndarray): Raw image data (n_samples, 784).
        add_bias (bool): Adds bias term (1) to each sample if True.
        normalize (bool): Normalizes pixel values to [0, 1] range if True.

    Returns:
        ndarray: Preprocessed dataset ready for training/testing.
    """
    if normalize:
        X = X / 255.0
    
    if add_bias:
        X = np.hstack((np.ones((X.shape[0], 1)), X))

    return X


def get_mnist_train_test(normalize=True, test_size=10000):
    """
    Loads MNIST data, preprocesses it, and splits into train/test sets.

    Args:
        normalize (bool): If True, normalize pixel values to [0, 1].
    
    Returns:
        X_train, X_test, y_train, y_test (ndarrays)
    """
    X, y = load_mnist()
    X_processed = preprocess_data(X, normalize=True, add_bias=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=10000, random_state=42
    )

    return X_train, X_test, y_train, y_test