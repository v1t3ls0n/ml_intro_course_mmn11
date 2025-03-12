# import numpy as np

def custom_confusion_matrix(y_true, y_pred, num_classes=10):
    """
    Compute confusion matrix from scratch.
    Args:
        y_true (array): True labels.
        y_pred (array): Predicted labels.
        num_classes (int): Number of unique classes.

    Returns:
        cm (ndarray): num_classes x num_classes confusion matrix.
                      Rows represent actual labels, columns represent predicted labels.
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        cm[actual, predicted] += 1
    return cm
