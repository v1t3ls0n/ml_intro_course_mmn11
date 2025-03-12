import numpy as np
import time
from core.logger.config import logger

class SoftmaxRegression:
    """
    Multinomial Logistic Regression (Softmax) for multi-class classification.
    Uses full-batch gradient descent on the cross-entropy loss.
    """

    def __init__(self, 
                 num_classes=10, 
                 max_iter=200, 
                 learning_rate=0.01):
        """
        Args:
            num_classes (int): Number of classes (10 for MNIST).
            max_iter (int): Maximum number of gradient descent iterations.
            learning_rate (float): Step size for gradient updates.
        """
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        # Will be initialized in fit() once X_train shape is known
        self.weights = None  # shape: (num_classes, n_features)

        # For tracking training progress
        self.loss_history = []
        self.training_runtime = None

    def _one_hot_encode(self, y):
        """
        Converts integer labels into one-hot vectors.
        e.g., 5 -> [0,0,0,0,0,1,0,0,0,0]
        """
        n_samples = y.shape[0]
        one_hot = np.zeros((n_samples, self.num_classes))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot

    def _softmax(self, logits):
        """
        Applies the softmax function row-wise.
        logits shape: (n_samples, num_classes)
        returns probabilities of same shape.
        """
        # For numerical stability, subtract max from each row
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        sums = np.sum(exps, axis=1, keepdims=True)
        return exps / sums

    def _cross_entropy_loss(self, probs, y_one_hot):
        """
        Computes cross-entropy loss:
            L = -1/N * sum( y_one_hot * log(probs) )
        """
        n_samples = y_one_hot.shape[0]
        # Clip probabilities to avoid log(0)
        eps = 1e-15
        log_probs = np.log(np.clip(probs, eps, 1 - eps))
        loss = -np.sum(y_one_hot * log_probs) / n_samples
        return loss

    def fit(self, X, y, 
            X_val=None, y_val=None, 
            X_test=None, y_test=None):
        """
        Trains the Softmax Regression model via batch gradient descent.
        Optionally tracks validation/test loss if provided.

        Args:
            X (ndarray): Training data of shape (n_samples, n_features).
            y (ndarray): True labels of shape (n_samples,).
            X_val, y_val (ndarray, optional): Validation data for monitoring.
            X_test, y_test (ndarray, optional): Test data for monitoring.
        """
        start_time = time.time()

        # One-hot encode the labels
        y_one_hot = self._one_hot_encode(y)

        # Initialize weights if not already
        n_features = X.shape[1]
        if self.weights is None:
            self.weights = np.zeros((self.num_classes, n_features))

        for iteration in range(self.max_iter):
            # Compute logits: shape (n_samples, num_classes)
            logits = X @ self.weights.T
            # Compute probabilities via softmax
            probs = self._softmax(logits)

            # Compute cross-entropy loss (for logging)
            loss = self._cross_entropy_loss(probs, y_one_hot)
            self.loss_history.append(loss)

            # Gradient of cross-entropy w.r.t. weights
            # grad shape: (num_classes, n_features)
            # (probs - y_one_hot) has shape (n_samples, num_classes)
            # X has shape (n_samples, n_features)
            # We'll do gradient per class:
            n_samples = X.shape[0]
            dW = (probs - y_one_hot).T @ X  # shape: (num_classes, n_features)
            dW /= n_samples

            # Update weights
            self.weights -= self.learning_rate * dW

            if (iteration + 1) % 10 == 0:
                logger.info(f"Iter {iteration+1}/{self.max_iter}, Loss: {loss:.4f}")

        self.training_runtime = time.time() - start_time
        logger.info(f"SoftmaxRegression training completed in {self.training_runtime:.2f} seconds.")

    def predict_proba(self, X):
        """
        Returns softmax probabilities for each sample.
        shape of return: (n_samples, num_classes)
        """
        logits = X @ self.weights.T
        return self._softmax(logits)

    def predict(self, X):
        """
        Returns predicted class labels for each sample.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
