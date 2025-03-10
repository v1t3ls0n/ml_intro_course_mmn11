import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix

# Enable inline plotting in Jupyter Notebook
%matplotlib inline

# -----------------------------
# 1. Data Loading and Preprocessing
# -----------------------------
def load_mnist():
    """
    Loads the MNIST dataset using fetch_openml.
    Returns:
        X: Array of shape (n_samples, 784) with pixel features.
        y: Array of shape (n_samples,) with digit labels.
    """
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int32)
    return X, y

def preprocess_data(X):
    """
    Normalizes pixel values to [0, 1] and adds a bias term.
    Each image (28x28 pixels) is flattened into a 785-dimensional vector,
    where the first element is set to 1 (bias).
    """
    X = X / 255.0
    # Add bias column (first column set to 1)
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    return X_bias

# -----------------------------
# 2. Custom Confusion Matrix Implementation
# -----------------------------
def custom_confusion_matrix(y_true, y_pred, num_classes=10):
    """
    Computes the confusion matrix from scratch.
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        num_classes: Number of classes (default 10 for MNIST).
    Returns:
        A (num_classes x num_classes) confusion matrix.
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# -----------------------------
# 3. Multi-Class Perceptron with Pocket Algorithm
# -----------------------------
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

# -----------------------------
# 4. Evaluation Functions
# -----------------------------
def evaluate_model(model, X, y):
    """
    Evaluates the trained model on data X with true labels y.
    Computes and prints:
      - Confusion matrix using a prebuilt function.
      - Confusion matrix using a custom implementation.
      - Overall accuracy.
      - Sensitivity (TPR) for each class.
    """
    y_pred = model.predict(X)
    # Prebuilt confusion matrix (from scikit-learn)
    prebuilt_cm = confusion_matrix(y, y_pred)
    # Custom confusion matrix implementation
    custom_cm = custom_confusion_matrix(y, y_pred, num_classes=model.num_classes)
    
    accuracy = np.trace(prebuilt_cm) / np.sum(prebuilt_cm)
    print("Prebuilt Confusion Matrix:\n", prebuilt_cm)
    print("Custom Confusion Matrix:\n", custom_cm)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
    # Compute sensitivity for each class: TPR = TP / (TP + FN)
    for cls in range(model.num_classes):
        TP = prebuilt_cm[cls, cls]
        FN = np.sum(prebuilt_cm[cls, :]) - TP
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        print(f"Sensitivity for digit {cls}: {tpr:.2f}")
    
    return prebuilt_cm, accuracy

def plot_confusion_matrix(cm):
    """
    Plots the confusion matrix as a heatmap.
    """
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted Digit")
    plt.ylabel("True Digit")
    plt.show()

# -----------------------------
# 5. Main Execution (Jupyter Notebook)
# -----------------------------
if __name__ == "__main__":
    # Load and preprocess the MNIST dataset.
    X, y = load_mnist()
    X = preprocess_data(X)
    
    # Split data: first 60,000 for training, last 10,000 for testing.
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    # Further split the training set to obtain a validation set for plotting loss curves.
    X_train_main, X_val = X_train[:-5000], X_train[-5000:]
    y_train_main, y_val = y_train[:-5000], y_train[-5000:]
    
    # Initialize and train the multi-class perceptron.
    mcp = MultiClassPerceptron(learning_rate=0.01, max_iter=1000)
    mcp.fit(X_train_main, y_train_main, X_val, y_val)
    
    # Evaluate the model on the test set.
    prebuilt_cm, accuracy = evaluate_model(mcp, X_test, y_test)
    
    # Plot the prebuilt confusion matrix.
    plot_confusion_matrix(prebuilt_cm)
    
    # Plot training and validation loss curves for the binary classifier distinguishing digit 0.
    train_losses, val_losses = mcp.loss_history[0]
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Training Loss", marker='o')
    if val_losses:
        plt.plot(val_losses, label="Validation Loss", marker='x')
    plt.xlabel("Iteration")
    plt.ylabel("Misclassification Count")
    plt.title("Loss Curves for Classifier (Digit 0 vs. Rest)")
    plt.legend()
    plt.grid(True)
    plt.show()
