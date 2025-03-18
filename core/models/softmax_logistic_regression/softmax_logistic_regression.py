import numpy as np
import time
from core.logger.config import logger

class SoftmaxRegression:
    """
    Multinomial Logistic Regression (Softmax) for multi-class classification.
    Uses full-batch gradient descent on cross-entropy loss.
    """

    def __init__(self, 
                 num_classes=10, 
                 max_iter=200, 
                 learning_rate=0.01):
        """
        Args:
            num_classes (int): Number of classes.
            max_iter (int): Maximum iterations for gradient descent.
            learning_rate (float): Step size for gradient updates.
        """
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        # Weight matrix (initialized in fit)
        self.weights = None  # shape: (num_classes, n_features)

        # Store training/test losses by class
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
        """
        n_samples = y.shape[0]
        one_hot = np.zeros((n_samples, self.num_classes))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot

    def _softmax(self, logits):
        """
        Applies the softmax function row-wise.
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
        eps = 1e-15
        log_probs = np.log(np.clip(probs, eps, 1 - eps))
        return -np.sum(y_one_hot * log_probs) / n_samples

    def fit(self, X, y, X_test=None, y_test=None):
        """
        Trains the Softmax Regression model via batch gradient descent.
        Optionally computes test loss if X_test,y_test are provided.
        """
        start_time = time.time()

        # One-hot encode labels
        y_one_hot = self._one_hot_encode(y)
        n_samples, n_features = X.shape

        # Initialize weights if not already
        if self.weights is None:
            self.weights = np.zeros((self.num_classes, n_features))

        # If test data provided, prepare one-hot as well
        if X_test is not None and y_test is not None:
            y_test_one_hot = self._one_hot_encode(y_test)
        else:
            y_test_one_hot = None

        for iteration in range(self.max_iter):
            # Forward pass
            logits = X @ self.weights.T
            probs = self._softmax(logits)
            train_loss = self._cross_entropy_loss(probs, y_one_hot)

            # Record the same train loss under each class (to mirror structure)
            for i in range(self.num_classes):
                self.loss_history[i]["train"].append(train_loss)

            # Optional test loss
            if X_test is not None and y_test_one_hot is not None:
                test_logits = X_test @ self.weights.T
                test_probs = self._softmax(test_logits)
                test_loss = self._cross_entropy_loss(test_probs, y_test_one_hot)
                for i in range(self.num_classes):
                    self.loss_history[i]["test"].append(test_loss)

            # Gradient descent step
            dW = (probs - y_one_hot).T @ X  # shape: (num_classes, n_features)
            dW /= n_samples
            self.weights -= self.learning_rate * dW

            if (iteration + 1) % 10 == 0:
                logger.info(f"Iter {iteration+1}/{self.max_iter}, Loss: {train_loss:.4f}")

        # Populate final tracking info
        total_iters = self.max_iter
        for i in range(self.num_classes):
            self.converged_iterations[i] = total_iters
            self.final_train_error[i] = self.loss_history[i]["train"][-1]
            if X_test is not None and y_test_one_hot is not None:
                self.final_test_error[i] = self.loss_history[i]["test"][-1]
            else:
                self.final_test_error[i] = None

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
