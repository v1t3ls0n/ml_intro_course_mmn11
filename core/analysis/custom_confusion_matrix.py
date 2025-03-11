import numpy as np

def custom_confusion_matrix(y_true, y_pred, num_classes=10):
    '''
    Computes confusion matrix from scratch.
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        num_classes (int): Number of classes
    Returns:
        ndarray: Confusion matrix of shape (num_classes, num_classes)
    '''
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm
