import numpy as np
import time
from core.logger.config import logger

class LinearRegression:
    """
    Multi-class classifier that uses linear regression (least squares) on one-hot labels.
    Trains via batch gradient descent.
    """

    def __init__(self, 
                 num_classes=10, 
                 max_iter=100, 
                 learning_rate=0.001):
        """
        Args:
            num_classes (int): Number of classes (e.g., 10 for MNIST).
            max_iter (int): Maximum number of gradient descent iterations.
            learning_rate (float): Step size for gradient updates.
        """
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        # Weights will be shape (num_classes, n_features)
        self.weights = None  

        # We'll store losses by class, in the same style as Softmax/Perceptron:
        self.loss_history = {
            i: {"train": [], "test": []} for i in range(num_classes)
        }

        self.converged_iterations = {}
        self.final_train_error = {}
        self.final_test_error = {}
        self.training_runtime = None

    def _one_hot_encode(self, y):
        """
        Converts integer labels into one-hot vectors.
        For label k, creates a vector of length num_classes with
        a 1 in position k and 0 elsewhere.
        """
        n_samples = y.shape[0]
        one_hot = np.zeros((n_samples, self.num_classes))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot

    def _mse_loss(self, preds, y_one_hot):
        """
        Mean Squared Error (MSE):
            MSE = 1/(2N) * sum( (preds - y_one_hot)^2 )
        or use 1/(N). We'll do 1/(2N) for convenience.
        """
        n_samples = y_one_hot.shape[0]
        # The factor 1/2 is optional. We'll include it to match standard MSE definition.
        return 0.5 * np.mean((preds - y_one_hot) ** 2)

    def fit(self, X, y, X_test=None, y_test=None):
        """
        Trains the Linear Regression classifier (one-hot approach) via
        batch gradient descent on the MSE loss.

        Args:
            X (ndarray): Training data of shape (n_samples, n_features).
            y (ndarray): True labels of shape (n_samples,).
            X_test (ndarray, optional): Test data for tracking test loss.
            y_test (ndarray, optional): Test labels for tracking test loss.
        """
        start_time = time.time()

        # One-hot encode the labels
        y_one_hot = self._one_hot_encode(y)
        n_samples, n_features = X.shape

        # Initialize weights if not already
        if self.weights is None:
            self.weights = np.zeros((self.num_classes, n_features))

        # Prepare test data if provided
        if X_test is not None and y_test is not None:
            y_test_one_hot = self._one_hot_encode(y_test)
        else:
            y_test_one_hot = None

        for iteration in range(self.max_iter):
            # Compute predictions: shape (n_samples, num_classes)
            preds = X @ self.weights.T

            # Compute training loss
            train_loss = self._mse_loss(preds, y_one_hot)

            # Record the same train loss under each class (like in Softmax)
            for i in range(self.num_classes):
                self.loss_history[i]["train"].append(train_loss)

            # Optionally compute test loss
            if X_test is not None and y_test_one_hot is not None:
                test_preds = X_test @ self.weights.T
                test_loss = self._mse_loss(test_preds, y_test_one_hot)
                for i in range(self.num_classes):
                    self.loss_history[i]["test"].append(test_loss)

            # Compute gradient of MSE w.r.t. weights
            # MSE gradient: (1/N)* ( (preds - y_one_hot).T @ X )
            # shape: (num_classes, n_features)
            diff = preds - y_one_hot              # shape (n_samples, num_classes)
            grad = (diff.T @ X) / n_samples       # shape (num_classes, n_features)

            # Gradient descent step
            self.weights -= self.learning_rate * grad

            # Optional logging every 10 iterations
            if (iteration + 1) % 10 == 0:
                logger.info(f"Iter {iteration+1}/{self.max_iter}, Train Loss: {train_loss:.4f}")

        # Populate final tracking info for each class
        for i in range(self.num_classes):
            self.converged_iterations[i] = self.max_iter
            self.final_train_error[i] = self.loss_history[i]["train"][-1]
            if X_test is not None and y_test_one_hot is not None:
                self.final_test_error[i] = self.loss_history[i]["test"][-1]
            else:
                self.final_test_error[i] = None

        self.training_runtime = time.time() - start_time
        logger.info(f"LinearRegressionClassifier training completed in {self.training_runtime:.2f} seconds.")

    def predict_scores(self, X):
        """
        Returns raw linear scores (no softmax).
        Shape: (n_samples, num_classes)
        """
        return X @ self.weights.T

    def predict(self, X):
        """
        Returns predicted class labels for each sample.
        We do an argmax over the raw linear outputs (scores).
        """
        scores = self.predict_scores(X)
        return np.argmax(scores, axis=1)
