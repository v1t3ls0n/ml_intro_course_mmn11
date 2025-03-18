import numpy as np
import pandas as pd
import time
from core.logger.config import logger

class LinearRegression:
    """
    Multi-class classifier that uses linear regression (least squares) on one-hot labels.
    Trains via batch gradient descent using an AdaGrad-style adaptive learning rate (if enabled).
    Supports early stopping if the training loss improvement is too small.
    """

    def __init__(self, 
                 num_classes=10, 
                 max_iter=100, 
                 learning_rate=0.001,
                 adaptive_lr=True,
                 early_stopping=False,
                 tol=1e-5,
                 patience=10):
        """
        Args:
            num_classes (int): Number of classes (e.g., 10 for MNIST).
            max_iter (int): Maximum number of gradient descent iterations.
            learning_rate (float): Base step size for gradient updates.
            adaptive_lr (bool): Flag to enable/disable adaptive learning rate. Default is True.
            early_stopping (bool): Flag to enable early stopping based on loss improvement. Default is False.
            tol (float): Minimum improvement in training loss required to reset patience. Default is 1e-5.
            patience (int): Number of consecutive iterations with insufficient improvement before stopping.
        """
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.adaptive_lr = adaptive_lr
        self.early_stopping = early_stopping
        self.tol = tol
        self.patience = patience

        # Weights will be shape (num_classes, d_features)
        self.weights = None  

        # We'll store losses by class
        self.loss_history = {
            i: {"train": [], "test": []} for i in range(num_classes)
        }

        self.converged_iterations = {}
        self.final_train_error = {}
        self.final_test_error = {}
        self.training_runtime = None

        # For AdaGrad
        self.G = None

        # We'll store iteration logs in a list of dicts, then convert to a DataFrame
        self.iter_logs = []

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
        n_samples, d_features = X.shape

        # Initialize weights if not already set
        if self.weights is None:
            # small random initialization
            self.weights = np.random.randn(self.num_classes, d_features) * 0.01

        # Initialize gradient accumulator for AdaGrad if adaptive learning rate is enabled
        if self.adaptive_lr and self.G is None:
            self.G = np.zeros_like(self.weights)

        # Prepare test data if provided
        if X_test is not None and y_test is not None:
            y_test_one_hot = self._one_hot_encode(y_test)
        else:
            y_test_one_hot = None

        epsilon = 1e-8       # Small constant for numerical stability
        lambda_reg = 0.01    # L2 regularization parameter
        max_grad_norm = 1.0  # Threshold for gradient clipping

        # Early stopping variables if enabled
        if self.early_stopping:
            best_loss = np.inf
            wait = 0

        for iteration in range(self.max_iter):
            # Compute predictions: shape (n_samples, num_classes)
            preds = X @ self.weights.T

            # Compute training loss
            train_loss = self._mse_loss(preds, y_one_hot)
            for i in range(self.num_classes):
                self.loss_history[i]["train"].append(train_loss)

            # Optionally compute test loss
            if X_test is not None and y_test_one_hot is not None:
                test_preds = X_test @ self.weights.T
                test_loss = self._mse_loss(test_preds, y_test_one_hot)
                for i in range(self.num_classes):
                    self.loss_history[i]["test"].append(test_loss)
            else:
                test_loss = None

            # Early stopping check if enabled
            if self.early_stopping:
                if best_loss - train_loss > self.tol:
                    best_loss = train_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        logger.info(f"Early stopping triggered at iteration {iteration+1} with training loss {train_loss:.6f}")
                        break

            # Compute gradient of MSE with respect to weights
            dW = (preds - y_one_hot).T @ X
            dW /= n_samples

            # gradient clipping
            grad_norm = np.linalg.norm(dW)
            if grad_norm > max_grad_norm:
                dW *= max_grad_norm / grad_norm

            # L2 regularization
            dW += lambda_reg * self.weights

            if self.adaptive_lr:
                # AdaGrad accumulator
                self.G += dW ** 2
                # adaptive update
                adaptive_dW = dW / (np.sqrt(self.G) + epsilon)
                self.weights -= self.learning_rate * adaptive_dW
            else:
                # standard GD update
                self.weights -= self.learning_rate * dW

            # iteration logging
            iter_data = {
                "iteration": iteration + 1,
                "train_loss": train_loss,
                "gradient_norm": grad_norm
            }
            if test_loss is not None:
                iter_data["test_loss"] = test_loss

            # log an average adaptive LR if in AdaGrad mode
            if self.adaptive_lr:
                avg_adaptive_lr = self.learning_rate / (np.mean(np.sqrt(self.G) + epsilon))
                iter_data["avg_adaptive_lr"] = avg_adaptive_lr

            self.iter_logs.append(iter_data)

            # logging every 100 iterations
            if (iteration + 1) % 100 == 0:
                logger.info(
                    f"Iter {iteration+1}/{self.max_iter}, Loss: {train_loss:.4f}, "
                    f"Gradient Norm: {grad_norm:.4f}, "
                    f"Avg Adaptive LR: {iter_data.get('avg_adaptive_lr', 'N/A')}"
                )

        self.training_runtime = time.time() - start_time
        logger.info(f"LinearRegressionClassifier training completed in {self.training_runtime:.2f} seconds.")

    def get_iteration_df(self):
        """
        Returns a pandas DataFrame with columns like:
          ['iteration', 'train_loss', 'test_loss', 'gradient_norm', 'avg_adaptive_lr']
        """
        return pd.DataFrame(self.iter_logs)

    def predict_scores(self, X):
        """
        Returns raw linear scores (no softmax).
        Shape: (n_samples, num_classes)
        """
        return X @ self.weights.T

    def predict(self, X):
        """
        Returns predicted class labels for each sample.
        """
        scores = self.predict_scores(X)
        return np.argmax(scores, axis=1)
