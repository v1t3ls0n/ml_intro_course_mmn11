# mnist_loader.py
from sklearn.datasets import fetch_openml
import numpy as np

def load_mnist(version=1):
    """
    Loads the MNIST dataset using scikit-learn's fetch_openml.
    
    Returns:
        X (ndarray): Array of pixel features (shape: samples x 784).
        y (ndarray): Digit labels (0-9).
    """
    try:
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        X, y = mnist['data'], mnist['target'].astype(int)
        return X, y
    except Exception as e:
        logger.info(f"Error loading MNIST dataset: {e}")
        return None, None
