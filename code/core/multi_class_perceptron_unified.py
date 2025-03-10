import numpy as np
from log.config import logger

class MultiClassPerceptron:
    """
    A unified multi-class perceptron that can operate in:
      - "clean" (no-pocket) mode, or
      - "pocket" mode.
    Controlled by the 'use_pocket' parameter in __init__.
    """

    def __init__(self, num_classes=10, max_iter=1000, use_pocket=True):
        """
        Args:
            num_classes (int): Number of classes (10 for MNIST).
            max_iter (int): Maximum number of iterations for the batch update loop.
            use_pocket (bool): If True, use the pocket algorithm; if False, no pocket logic.
        """
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.use_pocket = use_pocket

        # Each row corresponds to a weight vector for one binary classifier.
        self.weights = np.zeros((num_classes, 785))

        # Record loss history as a tuple (train_losses, val_losses) for each classifier.
        self.loss_history = {i: ([], []) for i in range(num_classes)}

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Trains the multi-class perceptron using one-vs-all strategy.
        For each class i, create a binary classification problem:
          +1 for class i, and -1 for all other classes.
        Optionally, validation data can be provided to track validation loss.

        Args:
            X (ndarray): Training data with bias, shape (n_samples, n_features).
            y (ndarray): True labels (digits 0..9), shape (n_samples,).
            X_val (ndarray, optional): Validation data with bias. Defaults to None.
            y_val (ndarray, optional): Validation labels. Defaults to None.
        """
        for cls in range(self.num_classes):
            logger.info(f"Training binary classifier for digit {cls}...")
            # Create binary labels: +1 for 'cls', -1 for all others
            binary_labels = np.where(y == cls, 1, -1)
            if X_val is not None and y_val is not None:
                binary_val_labels = np.where(y_val == cls, 1, -1)
            else:
                binary_val_labels = None

            best_w, train_losses, val_losses = self._train_binary(
                X, binary_labels, cls, X_val, binary_val_labels
            )
            self.weights[cls] = best_w
            self.loss_history[cls] = (train_losses, val_losses)

    def _train_binary(self, X, binary_labels, cls_idx, X_val=None, val_labels=None):
        """
        Trains a single binary perceptron (for digit `cls_idx`).
        If 'use_pocket' is True, uses pocket logic; otherwise, does "clean" updates only.

        Args:
            X (ndarray): Training data (with bias).
            binary_labels (ndarray): +1 / -1 labels for the current class.
            cls_idx (int): The digit index being trained (for logging).
            X_val (ndarray, optional): Validation data. Defaults to None.
            val_labels (ndarray, optional): Validation labels (+1 / -1). Defaults to None.

        Returns:
            final_w (ndarray): The final weight vector used (pocket or not).
            train_losses (list): Number of misclassifications after each iteration.
            val_losses (list): Validation misclassifications after each iteration (if X_val is provided).
        """
        n_samples, n_features = X.shape
        w = np.zeros(n_features)  # current weight vector

        # If using pocket, track the best w so far
        if self.use_pocket:
            pocket_w = w.copy()
            pocket_error = self._compute_error(X, binary_labels, w)

        # Initial error
        current_error = self._compute_error(X, binary_labels, w)
        train_losses = [current_error]
        val_losses = []

        # Initial validation error if provided
        if X_val is not None and val_labels is not None:
            val_error = self._compute_error(X_val, val_labels, w)
            val_losses.append(val_error)

        for t in range(self.max_iter):
            # Vectorized predictions
            preds = np.sign(X @ w)
            preds[preds == 0] = -1  # treat 0 as -1

            # Identify misclassified samples
            misclassified = (preds != binary_labels)
            num_misclassified = np.sum(misclassified)

            # If no misclassifications, we've perfectly separated or reached best so far
            if num_misclassified == 0:
                logger.info(f"Classifier for digit {cls_idx} converged after {t} iterations.")
                break

            # Batch update: sum of all misclassified samples
            update = np.sum(
                X[misclassified] * binary_labels[misclassified][:, None],
                axis=0
            )
            w += update

            # Compute current training error
            current_error = self._compute_error(X, binary_labels, w)
            train_losses.append(current_error)

            # Validation error if provided
            if X_val is not None and val_labels is not None:
                val_error = self._compute_error(X_val, val_labels, w)
                val_losses.append(val_error)

            # If pocket is enabled, check if this w is better
            if self.use_pocket:
                if current_error < pocket_error:
                    pocket_error = current_error
                    pocket_w = w.copy()

        # Return either the final w or the best pocket w
        final_w = pocket_w if (self.use_pocket) else w
        return final_w, train_losses, val_losses

    def _compute_error(self, X, labels, w):
        """
        Computes the number of misclassifications for weight vector w.
        """
        preds = np.sign(X @ w)
        preds[preds == 0] = -1
        return np.sum(preds != labels)

    def predict(self, X):
        """
        Predicts the digit label for each sample in X (with bias).
        For each sample, compute w_i^T x for all classes i, pick the max.
        """
        scores = X @ self.weights.T  # shape (n_samples, num_classes)
        return np.argmax(scores, axis=1)
