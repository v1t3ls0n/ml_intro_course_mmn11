# Data Loading and Preprocessing Functions
from imports import np, fetch_openml

def load_mnist():
    """
    Loads the MNIST dataset using fetch_openml.
    Returns:
        X: Array of shape (n_samples, 784) with pixel features.
        y: Array of shape (n_samples,) with digit labels.
    """
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int32)
    return X, y

def preprocess_data(X):
    """
    Normalizes pixel values to [0, 1] and adds a bias term.
    Each image (28x28 pixels) is flattened into a 785-dimensional vector,
    where the first element is set to 1 (bias).
    """
    X = X / 255.0
    # Add bias column (first column set to 1)
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    return X_bias
