# Custom Confusion Matrix Implementation
import numpy as np
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
