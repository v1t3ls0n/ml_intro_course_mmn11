import numpy as np
from core.logger.config import logger
import time

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

        # For each class i, we'll store: (train_losses, val_losses, test_losses)
        # train_losses[i] => list of iteration-level training errors for class i
        # val_losses[i]   => list of iteration-level validation errors (if X_val,y_val)
        # test_losses[i]  => list of iteration-level test errors (if X_test,y_test)
        self.loss_history = {
            i: {
                "train": [],
                "val":   [],
                "test":  []
            } for i in range(num_classes)
        }

        # Additional tracking
        self.converged_iterations = {}   # how many iterations each class used
        self.final_train_error = {}      # last train error for each class
        self.final_val_error = {}        # last val error (if val set) for each class
        self.final_test_error = {}       # last test error (if test set is provided)
        self.training_runtime = None
    def fit(self, X, y, 
            X_val=None, y_val=None, 
            X_test=None, y_test=None):
        """
        Trains the multi-class perceptron using one-vs-all strategy.
        For each class i, create a binary classification problem:
          +1 for class i, and -1 for all other classes.

        Optionally, validation data (X_val, y_val) can be provided to track val error,
        and test data (X_test, y_test) to track test error per iteration (for real "train vs. test" curves).

        After training, you can retrieve iteration-level errors per class from:
          self.loss_history[i]["train"], self.loss_history[i]["val"], self.loss_history[i]["test"]
        and plot them with `plot_history`.
        """
        training_start_time = time.time()

        for cls in range(self.num_classes):
            logger.info(f"Training binary classifier for digit {cls}...")
            # Create binary labels: +1 for 'cls', -1 for all others
            binary_labels = np.where(y == cls, 1, -1)

            # Validation labels, if any
            if X_val is not None and y_val is not None:
                binary_val_labels = np.where(y_val == cls, 1, -1)
            else:
                binary_val_labels = None

            # Test labels, if any
            if X_test is not None and y_test is not None:
                binary_test_labels = np.where(y_test == cls, 1, -1)
            else:
                binary_test_labels = None

            (best_w, 
             train_losses, 
             val_losses, 
             test_losses, 
             iteration_count) = self._train_binary(
                X, 
                binary_labels, 
                cls, 
                X_val, 
                binary_val_labels,
                X_test,
                binary_test_labels
            )

            # Store final weights
            self.weights[cls] = best_w

            # Store iteration-level losses
            self.loss_history[cls]["train"] = train_losses
            self.loss_history[cls]["val"]   = val_losses
            self.loss_history[cls]["test"]  = test_losses

            # Record iteration count & final errors
            self.converged_iterations[cls] = iteration_count
            self.final_train_error[cls]    = train_losses[-1] if train_losses else None
            self.final_val_error[cls]      = val_losses[-1]   if val_losses else None
            self.final_test_error[cls]     = test_losses[-1]  if test_losses else None
            self.training_runtime = time.time() - training_start_time 

    def _train_binary(self, 
                      X, binary_labels, cls_idx, 
                      X_val=None, val_labels=None,
                      X_test=None, test_labels=None):
        """
        Trains a single binary perceptron (for digit `cls_idx`).
        If 'use_pocket' is True, uses pocket logic; otherwise, does "clean" updates only.

        We record iteration-level train error, and optionally val/test error if provided.

        Returns:
            final_w (ndarray): The final weight vector used (pocket or not).
            train_losses (list): iteration-level training error (# misclassifications).
            val_losses   (list): iteration-level validation error (if X_val,y_val).
            test_losses  (list): iteration-level test error (if X_test,y_test).
            iteration_count (int): how many iterations used before convergence or max_iter.
        """
        n_samples, n_features = X.shape
        w = np.zeros(n_features)  # current weight vector

        # If using pocket, track the best w so far
        if self.use_pocket:
            pocket_w = w.copy()
            pocket_error = self._compute_error(X, binary_labels, w)

        # Compute initial errors
        current_train_err = self._compute_error(X, binary_labels, w)
        train_losses = [current_train_err]

        val_losses  = []
        test_losses = []

        # If we have val data, compute initial val error
        if X_val is not None and val_labels is not None:
            val_error = self._compute_error(X_val, val_labels, w)
            val_losses.append(val_error)

        # If we have test data, compute initial test error
        if X_test is not None and test_labels is not None:
            test_error = self._compute_error(X_test, test_labels, w)
            test_losses.append(test_error)

        iteration_count = 0

        for t in range(self.max_iter):
            iteration_count = t + 1
            preds = np.sign(X @ w)
            preds[preds == 0] = -1  # treat 0 as -1

            misclassified = (preds != binary_labels)
            num_misclassified = np.sum(misclassified)

            # If no misclassifications => perfect separation
            if num_misclassified == 0:
                logger.info(f"Classifier for digit {cls_idx} converged after {iteration_count} iterations.")
                break

            # Perform batch update on all misclassified samples
            update = np.sum(X[misclassified] * binary_labels[misclassified][:, None], axis=0)
            w += update

            # Compute current training error
            current_train_err = self._compute_error(X, binary_labels, w)
            train_losses.append(current_train_err)

            # Validation error
            if X_val is not None and val_labels is not None:
                val_err = self._compute_error(X_val, val_labels, w)
                val_losses.append(val_err)

            # Test error
            if X_test is not None and test_labels is not None:
                test_err = self._compute_error(X_test, test_labels, w)
                test_losses.append(test_err)

            # If pocket is enabled, check if this w is better
            if self.use_pocket:
                if current_train_err < pocket_error:
                    pocket_error = current_train_err
                    pocket_w = w.copy()

        # Return either the final w or the best pocket w
        final_w = pocket_w if self.use_pocket else w

        return final_w, train_losses, val_losses, test_losses, iteration_count

    def _compute_error(self, X, labels, w):
        """
        Computes the number of misclassifications given weight vector w.
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