import numpy as np
import time
from core.logger.config import logger

class SoftmaxRegression:
    """
    Multinomial Logistic Regression (Softmax) for multi-class classification.
    Uses full-batch gradient descent on cross-entropy loss with an AdaGrad-style adaptive learning rate (if enabled).
    Supports early stopping if the training loss improvement is too small.
    """

    def __init__(self, 
                 num_classes=10, 
                 max_iter=200, 
                 learning_rate=0.01,
                 adaptive_lr=True,
                 early_stopping=True,
                 tol=1e-5,
                 patience=10):
        """
        Args:
            num_classes (int): Number of classes.
            max_iter (int): Maximum iterations for gradient descent.
            learning_rate (float): Base step size for gradient updates.
            adaptive_lr (bool): Flag to enable/disable AdaGrad-style adaptive learning rate. Default is True.
            early_stopping (bool): Flag to enable early stopping based on loss improvement. Default is True.
            tol (float): Minimum improvement in training loss to qualify as progress. Default is 1e-5.
            patience (int): Number of iterations with insufficient improvement before stopping. Default is 10.
        """
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.adaptive_lr = adaptive_lr
        self.early_stopping = early_stopping
        self.tol = tol
        self.patience = patience

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
        If adaptive_lr is enabled, uses an AdaGrad-style adaptive learning rate.
        If early_stopping is enabled, training stops when progress in loss improvement is too small.
        Optionally computes test loss if X_test and y_test are provided.
        """
        start_time = time.time()

        # One-hot encode labels
        y_one_hot = self._one_hot_encode(y)
        n_samples, n_features = X.shape

        # Initialize weights if not already set
        if self.weights is None:
            self.weights = np.zeros((self.num_classes, n_features))

        # Initialize gradient accumulator for AdaGrad if adaptive learning rate is enabled
        if self.adaptive_lr:
            if not hasattr(self, 'G'):
                self.G = np.zeros_like(self.weights)

        # Prepare one-hot test labels if test data is provided
        if X_test is not None and y_test is not None:
            y_test_one_hot = self._one_hot_encode(y_test)
        else:
            y_test_one_hot = None

        epsilon = 1e-8  # Small constant for numerical stability

        # Variables for early stopping
        if self.early_stopping:
            best_loss = np.inf
            wait = 0

        for iteration in range(self.max_iter):
            # Forward pass: compute logits, probabilities, and training loss
            logits = X @ self.weights.T
            probs = self._softmax(logits)
            train_loss = self._cross_entropy_loss(probs, y_one_hot)

            # Record training loss for each class
            for i in range(self.num_classes):
                self.loss_history[i]["train"].append(train_loss)

            # Optionally compute and record test loss
            if X_test is not None and y_test_one_hot is not None:
                test_logits = X_test @ self.weights.T
                test_probs = self._softmax(test_logits)
                test_loss = self._cross_entropy_loss(test_probs, y_test_one_hot)
                for i in range(self.num_classes):
                    self.loss_history[i]["test"].append(test_loss)

            # Check early stopping condition if enabled
            if self.early_stopping:
                if best_loss - train_loss > self.tol:
                    best_loss = train_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        logger.info(f"Early stopping triggered at iteration {iteration+1} with training loss {train_loss:.6f}")
                        break

            # Gradient computation
            dW = (probs - y_one_hot).T @ X  # shape: (num_classes, n_features)
            dW /= n_samples

            if self.adaptive_lr:
                # Accumulate squared gradients (AdaGrad accumulator)
                self.G += dW ** 2

                # Compute adaptive gradient update: scale by inverse square root of accumulated gradients
                adaptive_dW = dW / (np.sqrt(self.G) + epsilon)

                # Update weights using the adaptive gradient update
                self.weights -= self.learning_rate * adaptive_dW
            else:
                # Standard gradient descent update
                self.weights -= self.learning_rate * dW

            if (iteration + 1) % 100 == 0:
                if self.adaptive_lr:
                    avg_adaptive_lr = self.learning_rate / (np.mean(np.sqrt(self.G) + epsilon))
                    logger.info(f"Iter {iteration+1}/{self.max_iter}, Loss: {train_loss:.4f}, Avg Adaptive LR: {avg_adaptive_lr:.6f}")
                else:
                    logger.info(f"Iter {iteration+1}/{self.max_iter}, Loss: {train_loss:.4f}")

        # Populate final tracking info; use the iteration count at exit
        total_iters = iteration + 1
        for i in range(self.num_classes):
            self.converged_iterations[i] = total_iters
            self.final_train_error[i] = self.loss_history[i]["train"][-1]
            if X_test is not None and y_test_one_hot is not None and self.loss_history[i]["test"]:
                self.final_test_error[i] = self.loss_history[i]["test"][-1]
            else:
                self.final_test_error[i] = None

        self.training_runtime = time.time() - start_time
        logger.info(f"SoftmaxRegression training completed in {self.training_runtime:.2f} seconds.")

    def predict_proba(self, X):
        """
        Returns softmax probabilities for each sample.
        Shape: (n_samples, num_classes)
        """
        logits = X @ self.weights.T
        return self._softmax(logits)

    def predict(self, X):
        """
        Returns predicted class labels for each sample.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
