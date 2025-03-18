import numpy as np
from core.logger.config import logger
import time

class MultiClassPerceptron:
    """
    Multi-class perceptron using one-vs-all.
    Can use pocket or clean updates (toggled by 'use_pocket').
    """

    def __init__(self, num_classes=10, max_iter=1000, use_pocket=True):
        """
        Args:
            num_classes (int): Number of classes.
            max_iter (int): Max iterations for batch update.
            use_pocket (bool): Use pocket logic if True.
        """
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.use_pocket = use_pocket

        # Weight matrix: each row is a class-specific weight vector
        self.weights = np.zeros((num_classes, 785))

        # Store training/test losses by class
        self.loss_history = {
            i: {"train": [], "test": []} for i in range(num_classes)
        }

        self.converged_iterations = {}
        self.final_train_error = {}
        self.final_test_error = {}
        self.training_runtime = None

    def fit(self, X, y, X_test=None, y_test=None):
        """
        Trains a one-vs-all set of binary perceptrons.
        Optionally tracks test losses if X_test,y_test are given.
        """
        training_start_time = time.time()

        for cls in range(self.num_classes):
            logger.info(f"Training for digit {cls}...")
            binary_labels = np.where(y == cls, 1, -1)

            # Prepare test labels if test data is provided
            binary_test_labels = np.where(y_test == cls, 1, -1) if (X_test is not None and y_test is not None) else None

            best_w, train_losses, test_losses, iters = self._train_binary(
                X, binary_labels, cls, X_test, binary_test_labels
            )

            self.weights[cls] = best_w
            self.loss_history[cls]["train"] = train_losses
            self.loss_history[cls]["test"] = test_losses
            self.converged_iterations[cls] = iters
            self.final_train_error[cls] = train_losses[-1] if train_losses else None
            self.final_test_error[cls] = test_losses[-1] if test_losses else None

        self.training_runtime = time.time() - training_start_time

    def _train_binary(self, X, y, cls_idx, X_test=None, y_test=None):
        """
        Trains a single binary perceptron for one class.
        Returns final/best weights and iteration-level error histories.
        """
        w = np.zeros(X.shape[1])
        if self.use_pocket:
            pocket_w = w.copy()
            pocket_err = self._compute_error(X, y, w)

        train_err = self._compute_error(X, y, w)
        train_losses = [train_err]
        test_losses = []

        if X_test is not None and y_test is not None:
            test_err = self._compute_error(X_test, y_test, w)
            test_losses.append(test_err)

        for t in range(self.max_iter):
            preds = np.sign(X @ w)
            preds[preds == 0] = -1
            misclassified = (preds != y)

            # Stop if perfect separation
            if not np.any(misclassified):
                logger.info(f"Digit {cls_idx} converged at iteration {t+1}.")
                break

            # Update with all misclassified samples
            w += np.sum(X[misclassified] * y[misclassified, None], axis=0)

            train_err = self._compute_error(X, y, w)
            train_losses.append(train_err)

            if X_test is not None and y_test is not None:
                test_err = self._compute_error(X_test, y_test, w)
                test_losses.append(test_err)

            # Pocket logic
            if self.use_pocket and train_err < pocket_err:
                pocket_err = train_err
                pocket_w = w.copy()

        final_w = pocket_w if self.use_pocket else w
        return final_w, train_losses, test_losses, len(train_losses)

    def _compute_error(self, X, y, w):
        """
        Returns the number of misclassified samples.
        """
        preds = np.sign(X @ w)
        preds[preds == 0] = -1
        return np.sum(preds != y)

    def predict(self, X):
        """
        Predicts class by choosing the weight vector with max activation.
        """
        scores = X @ self.weights.T
        return np.argmax(scores, axis=1)
