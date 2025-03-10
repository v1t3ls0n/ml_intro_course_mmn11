import numpy as np

class MultiClassPerceptron:
    def __init__(self, num_classes=10, max_iter=1000):
        """
        Multi-class perceptron (vectorized batch update) WITHOUT an explicit learning rate.
        num_classes: Number of classes (10 for MNIST).
        max_iter: Maximum number of iterations for the batch update loop.
        """
        self.num_classes = num_classes
        self.max_iter = max_iter
        # Each row corresponds to a weight vector for one binary classifier.
        self.weights = np.zeros((num_classes, 785))
        # Record loss history as a tuple (train_losses, val_losses) for each classifier.
        self.loss_history = {i: ([], []) for i in range(num_classes)}

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Trains the multi-class perceptron using one-vs-all strategy, without a learning rate.
        For each class i, we create a binary classification problem:
          +1 for class i, and -1 for all other classes.
        Optionally, validation data can be provided to track validation loss.
        """
        for cls in range(self.num_classes):
            print(f"Training binary classifier for digit {cls}...")
            # Create binary labels: +1 for current class, -1 for all others
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
        Vectorized batch update training for a single binary perceptron,
        using a pocket algorithm. No learning rate is used (equivalent to eta=1).
        """
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        pocket_w = w.copy()

        # Initial error
        pocket_error = self._compute_error(X, binary_labels, w)
        train_losses = [pocket_error]
        val_losses = []
        
        # Initial validation error if provided
        if X_val is not None and val_labels is not None:
            current_val_loss = self._compute_error(X_val, val_labels, w)
            val_losses.append(current_val_loss)

        for t in range(self.max_iter):
            # Vectorized predictions for all samples
            preds = np.sign(X @ w)
            preds[preds == 0] = -1  # treat 0 as -1

            # Identify misclassified samples
            misclassified = (preds != binary_labels)
            num_misclassified = np.sum(misclassified)

            # If no misclassifications, we have perfect separation or best so far
            if num_misclassified == 0:
                print(f"Classifier for digit {cls_idx} converged after {t} iterations.")
                break

            # Batch update: sum of all misclassified samples
            # w <- w + Σ( x_i * y_i ) for i in misclassified
            update = np.sum(
                X[misclassified] * binary_labels[misclassified][:, None],
                axis=0
            )
            w += update

            # Compute current training error
            current_error = self._compute_error(X, binary_labels, w)
            train_losses.append(current_error)

            # Check validation error if provided
            if X_val is not None and val_labels is not None:
                current_val_loss = self._compute_error(X_val, val_labels, w)
                val_losses.append(current_val_loss)

            # Pocket algorithm: keep the best weight vector found so far
            if current_error < pocket_error:
                pocket_error = current_error
                pocket_w = w.copy()

        return pocket_w, train_losses, val_losses

    def _compute_error(self, X, labels, w):
        """
        Counts the number of misclassifications given weight vector w.
        """
        preds = np.sign(X @ w)
        preds[preds == 0] = -1
        return np.sum(preds != labels)

    def predict(self, X):
        """
        Predicts the digit label for each sample.
        For each sample, compute the score w_i^T x for all classes and pick the max.
        """
        scores = X @ self.weights.T
        return np.argmax(scores, axis=1)
