# Multi-Class Perceptron with Pocket Algorithm
from imports import np
class MultiClassPerceptron:
    def __init__(self, num_classes=10, learning_rate=0.01, max_iter=1000):
        """
        Initializes the multi-class perceptron.
        Args:
            num_classes: Number of classes (10 for MNIST).
            learning_rate: Learning rate for weight updates.
            max_iter: Maximum number of iterations per binary classifier.
        """
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        # Each row corresponds to the weight vector for one binary classifier.
        self.weights = np.zeros((num_classes, 785))
        # Record loss history as a tuple (train_losses, val_losses) for each classifier.
        self.loss_history = {i: ([], []) for i in range(num_classes)}
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Trains the multi-class perceptron using the one-vs-all strategy.
        For each class i, creates a binary classification problem:
           +1 for examples of class i, and -1 for all others.
        Optionally, validation data can be provided to record loss curves.
        """
        for cls in range(self.num_classes):
            print(f"Training binary classifier for digit {cls}...")
            # Create binary labels: +1 for current class, -1 for all others.
            binary_labels = np.where(y == cls, 1, -1)
            if X_val is not None and y_val is not None:
                binary_val_labels = np.where(y_val == cls, 1, -1)
            else:
                binary_val_labels = None
            best_w, train_losses, val_losses = self._train_binary(X, binary_labels, cls, X_val, binary_val_labels)
            self.weights[cls] = best_w
            self.loss_history[cls] = (train_losses, val_losses)
    
    def _train_binary(self, X, binary_labels, cls_idx, X_val=None, val_labels=None):
        """
        Trains a binary perceptron for one class using the pocket algorithm.
        Args:
            X: Training data (with bias).
            binary_labels: Binary labels (+1 / -1) for the current classifier.
            cls_idx: Current class index (for reporting).
            X_val, val_labels: Optional validation data for loss tracking.
        Returns:
            pocket_w: Best weight vector found (with lowest training error).
            train_losses: List of training loss values per update.
            val_losses: List of validation loss values per update (if provided).
        """
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        pocket_w = w.copy()
        pocket_error = self._compute_error(X, binary_labels, w)
        train_losses = [pocket_error]
        val_losses = []
        if X_val is not None and val_labels is not None:
            current_val_loss = self._compute_error(X_val, val_labels, w)
            val_losses.append(current_val_loss)
        
        for t in range(self.max_iter):
            misclassified = False
            # Update on the first misclassified example in each iteration.
            for i in range(n_samples):
                pred = np.sign(np.dot(w, X[i]))
                if pred == 0:
                    pred = -1  # Treat zero as -1.
                if pred != binary_labels[i]:
                    w = w + self.learning_rate * binary_labels[i] * X[i]
                    misclassified = True
                    current_error = self._compute_error(X, binary_labels, w)
                    train_losses.append(current_error)
                    if X_val is not None and val_labels is not None:
                        current_val_loss = self._compute_error(X_val, val_labels, w)
                        val_losses.append(current_val_loss)
                    # Update pocket if current error is lower.
                    if current_error < pocket_error:
                        pocket_error = current_error
                        pocket_w = w.copy()
                    break  # Process one misclassified example per iteration.
            if not misclassified:
                print(f"Classifier for digit {cls_idx} converged after {t} iterations.")
                break
        return pocket_w, train_losses, val_losses
    
    def _compute_error(self, X, labels, w):
        """
        Computes the number of misclassifications given weights w.
        """
        preds = np.sign(np.dot(X, w))
        preds[preds == 0] = -1
        return np.sum(preds != labels)
    
    def predict(self, X):
        """
        Predicts the digit label for each input instance.
        For each sample, computes the confidence scores (w_i^T x) for all classes
        and returns the class with the highest score.
        """
        scores = np.dot(X, self.weights.T)
        return np.argmax(scores, axis=1)
