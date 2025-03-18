import numpy as np
import pandas as pd
import time
from core.logger.config import logger

class SoftmaxRegression:
    """
    Multinomial Logistic Regression (Softmax) for multi-class classification.
    Uses full-batch gradient descent on cross-entropy loss with an AdaGrad-style adaptive learning rate (if enabled).
    Supports early stopping if the training loss improvement is too small.
    Also implements improved weight initialization and learning rate scheduling.
    """

    def __init__(self, 
                 num_classes=10, 
                 max_iter=200, 
                 learning_rate=0.01,
                 lr_decay=0.0,
                 adaptive_lr=True,
                 early_stopping=True,
                 tol=1e-5,
                 patience=10):
        """
        Args:
            num_classes (int): Number of classes.
            max_iter (int): Maximum iterations for gradient descent.
            learning_rate (float): Base step size for gradient updates.
            lr_decay (float): Decay rate for learning rate scheduling.
                              The effective learning rate at iteration t is 
                              learning_rate / (1 + lr_decay * t).
            adaptive_lr (bool): Flag to enable/disable AdaGrad-style adaptive learning rate. Default is True.
            early_stopping (bool): Flag to enable early stopping based on loss improvement. Default is True.
            tol (float): Minimum improvement in training loss to qualify as progress. Default is 1e-5.
            patience (int): Number of iterations with insufficient improvement before stopping. Default is 10.
        """
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.adaptive_lr = adaptive_lr
        self.early_stopping = early_stopping
        self.tol = tol
        self.patience = patience

        # Weight matrix (initialized in fit)
        self.weights = None  # shape: (num_classes, d_features)

        # Store training/test losses by class
        self.loss_history = {
            i: {"train": [], "test": []} for i in range(num_classes)
        }
        self.converged_iterations = {}
        self.final_train_error = {}
        self.final_test_error = {}
        self.training_runtime = None

        # We'll store iteration-level logs in a list of dicts, then convert to DataFrame
        self.iter_logs = []  # each element is {"iteration": i, "train_loss": ..., "lr": ...}

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
        n_samples, d_features = X.shape

        # Improved weight initialization: small random values (e.g., from a normal distribution)
        if self.weights is None:
            delta = np.sqrt(1 / d_features)  # Small value for initialization
            self.weights = np.random.standard_normal(self.num_classes, d_features)  * delta # Optional: standard normal initialization

            # old initialization method
            # self.weights = np.random.randn(self.num_classes, d_features) * 0.01 # Optional: small random initialization


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
            # Learning rate scheduling: effective_lr decays over iterations
            effective_lr = self.learning_rate / (1 + self.lr_decay * iteration)

            # Forward pass: compute logits, probabilities, and training loss
            logits = X @ self.weights.T
            probs = self._softmax(logits)
            train_loss = self._cross_entropy_loss(probs, y_one_hot)

            # Record training loss for each class (same loss for all classes)
            for i in range(self.num_classes):
                self.loss_history[i]["train"].append(train_loss)

            # Optionally compute and record test loss
            if X_test is not None and y_test_one_hot is not None:
                test_logits = X_test @ self.weights.T
                test_probs = self._softmax(test_logits)
                test_loss = self._cross_entropy_loss(test_probs, y_test_one_hot)
                for i in range(self.num_classes):
                    self.loss_history[i]["test"].append(test_loss)
            else:
                test_loss = None

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

            # Gradient computation: fully vectorized
            dW = (probs - y_one_hot).T @ X
            dW /= n_samples

            if self.adaptive_lr:
                # Accumulate squared gradients (AdaGrad accumulator)
                self.G += dW ** 2
                # Compute adaptive gradient update: scale by inverse square root of accumulated gradients
                adaptive_dW = dW / (np.sqrt(self.G) + epsilon)
                # Update weights using effective learning rate
                self.weights -= effective_lr * adaptive_dW
            else:
                # Standard gradient descent update with scheduled learning rate
                self.weights -= effective_lr * dW

            # Logging iteration data in self.iter_logs
            iteration_data = {
                "iteration": iteration + 1,
                "train_loss": train_loss,
                "effective_lr": effective_lr
            }
            # If test_loss is computed
            if test_loss is not None:
                iteration_data["test_loss"] = test_loss

            # If using AdaGrad, log an "avg_adaptive_lr"
            if self.adaptive_lr:
                # avg_adaptive_lr is the ratio of effective_lr to the mean of sqrt(G)
                avg_adaptive_lr = effective_lr / (np.mean(np.sqrt(self.G) + epsilon))
                iteration_data["avg_adaptive_lr"] = avg_adaptive_lr

            self.iter_logs.append(iteration_data)

            # Logging every 100 iterations
            if iteration % 10 == 0:
                if self.adaptive_lr:
                    logger.info(
                        f"Iter {iteration+1}/{self.max_iter}, "
                        f"Loss: {train_loss:.4f}, "
                        f"Avg Adaptive LR: {iteration_data['avg_adaptive_lr']:.6f}"
                    )
                else:
                    logger.info(
                        f"Iter {iteration+1}/{self.max_iter}, Loss: {train_loss:.4f}"
                    )

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

    def get_iteration_df(self):
        """
        Returns a pandas DataFrame of iteration-level logs, including columns:
          ['iteration', 'train_loss', 'test_loss', 'effective_lr', 'avg_adaptive_lr']
        (Some columns may be missing if not used.)
        """
        return pd.DataFrame(self.iter_logs)

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
