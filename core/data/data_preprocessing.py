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

def one_hot_encode(labels, num_classes=10):
    """
    Converts integer digit labels into one-hot vectors.

    Example: label 5 -> [0,0,0,0,0,1,0,0,0,0] (for num_classes=10)

    Args:
        labels (ndarray): Array of integer labels (shape: (n_samples,)).
        num_classes (int): Number of classes (e.g., 10 for MNIST digits 0..9).

    Returns:
        ndarray: One-hot encoded labels of shape (n_samples, num_classes).
    """
    n_samples = labels.shape[0]
    one_hot = np.zeros((n_samples, num_classes), dtype=np.float32)
    one_hot[np.arange(n_samples), labels] = 1.0
    return one_hot

def get_mnist_train_test(normalize=True, test_size=10000):
    """
    Loads MNIST data, preprocesses it, and splits into train/test sets.

    Args:
        normalize (bool): If True, normalize pixel values to [0, 1].
        test_size (int): Number of samples to allocate to the test set.
    
    Returns:
        X_train, X_test, y_train, y_test (ndarrays)
    """
    X, y = load_mnist()
    X_processed = preprocess_data(X, normalize=normalize, add_bias=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test
