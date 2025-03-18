import numpy as np
import time
from core.logger.config import logger

class LinearRegression:
    """
    Multi-class classifier that uses linear regression (least squares) on one-hot labels.
    Trains via batch gradient descent using an AdaGrad-style adaptive learning rate.
    """

    def __init__(self, 
                 num_classes=10, 
                 max_iter=100, 
                 learning_rate=0.001):
        """
        Args:
            num_classes (int): Number of classes (e.g., 10 for MNIST).
            max_iter (int): Maximum number of gradient descent iterations.
            learning_rate (float): Base step size for gradient updates.
        """
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        # Weights will be shape (num_classes, n_features)
        self.weights = None  

        # We'll store losses by class
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
            MSE = 1/(N) * sum( (preds - y_one_hot)^2 )
        """
        return np.mean((preds - y_one_hot) ** 2)

    def fit(self, X, y, X_test=None, y_test=None):
        start_time = time.time()

        # One-hot encode the labels
        y_one_hot = self._one_hot_encode(y)
        n_samples, n_features = X.shape

        # Initialize weights if not already set
        if self.weights is None:
            self.weights = np.zeros((self.num_classes, n_features))

        # Initialize gradient accumulator for AdaGrad
        if not hasattr(self, 'G'):
            self.G = np.zeros_like(self.weights)

        # Prepare test data if provided
        if X_test is not None and y_test is not None:
            y_test_one_hot = self._one_hot_encode(y_test)
        else:
            y_test_one_hot = None

        epsilon = 1e-8       # Small constant for numerical stability
        lambda_reg = 0.01    # L2 regularization parameter
        max_grad_norm = 1.0  # Threshold for gradient clipping

        for iteration in range(self.max_iter):
            # Compute predictions: shape (n_samples, num_classes)
            preds = X @ self.weights.T

            # Compute training loss and record it
            train_loss = self._mse_loss(preds, y_one_hot)
            for i in range(self.num_classes):
                self.loss_history[i]["train"].append(train_loss)

            # Optionally compute and record test loss
            if X_test is not None and y_test_one_hot is not None:
                test_preds = X_test @ self.weights.T
                test_loss = self._mse_loss(test_preds, y_test_one_hot)
                for i in range(self.num_classes):
                    self.loss_history[i]["test"].append(test_loss)

            # Compute gradient of MSE with respect to weights
            dW = (preds - y_one_hot).T @ X
            dW /= n_samples

            # Compute gradient norm for possible clipping
            grad_norm = np.linalg.norm(dW)
            if grad_norm > max_grad_norm:
                dW *= max_grad_norm / grad_norm

            # Add L2 regularization term
            dW += lambda_reg * self.weights

            # Accumulate squared gradients (AdaGrad accumulator)
            self.G += dW ** 2

            # Compute the adaptive gradient update
            adaptive_dW = dW / (np.sqrt(self.G) + epsilon)

            # Update weights using the adaptive gradient update
            self.weights -= self.learning_rate * adaptive_dW

            # Check for NaN or Inf in weights
            if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)):
                logger.error("Weights contain NaN or Inf values. Stopping training.")
                break

            if (iteration + 1) % 100 == 0:
                avg_adaptive_lr = self.learning_rate / np.mean(np.sqrt(self.G) + epsilon)
                logger.info(f"Iter {iteration+1}/{self.max_iter}, Loss: {train_loss:.4f}, "
                            f"Gradient Norm: {grad_norm:.4f}, Avg Adaptive LR: {avg_adaptive_lr:.6f}")

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
