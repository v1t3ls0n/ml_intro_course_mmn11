# Evaluation Functions
# Imports and setup
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from logger.config import logger
from core.analysis.custom_confusion_matrix import custom_confusion_matrix



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
    accuracy = np.trace(prebuilt_cm) / np.sum(prebuilt_cm)
    logger.info("Prebuilt Confusion Matrix:\n", prebuilt_cm)


    # Custom confusion matrix implementation
    custom_cm = custom_confusion_matrix(y, y_pred, num_classes=model.num_classes)
    logger.info("Custom Confusion Matrix:\n", custom_cm)



    logger.info(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
    # Compute sensitivity for each class: TPR = TP / (TP + FN)
    for cls in range(model.num_classes):
        TP = prebuilt_cm[cls, cls]
        FN = np.sum(prebuilt_cm[cls, :]) - TP
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        logger.info(f"Sensitivity for digit {cls}: {tpr:.2f}")
    
    return prebuilt_cm, accuracy


def analyze_confusion_matrix(cm):
    """
    Computes and prints additional metrics beyond TPR:
      - TNR (True Negative Rate) for each class.
      - Optionally: precision, F1, or any other analysis you want.
    Args:
        cm (ndarray): Confusion matrix (num_classes x num_classes).
    """
    num_classes = cm.shape[0]
    total_samples = np.sum(cm)

    logger.info("\n=== Advanced Analysis ===")
    for cls in range(num_classes):
        TP = cm[cls, cls]
        FN = np.sum(cm[cls, :]) - TP
        FP = np.sum(cm[:, cls]) - TP
        TN = total_samples - (TP + FP + FN)

        # True Negative Rate (TNR)
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        
        logger.info(f"Class {cls}: TNR = {TNR:.2f}  (TN={TN}, FP={FP})")
    logger.info("=== End of Advanced Analysis ===\n")


def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None):
    """
    Plots the confusion matrix as a heatmap.
    
    Args:
        cm (ndarray): Confusion matrix (shape: (num_classes, num_classes)).
        title (str): Title of the plot.
        save_path (str, optional): If provided, file path to save the figure 
                                   (e.g. "../outputs/conf_mat.png").
    """
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted Digit")
    plt.ylabel("True Digit")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix_annotated(cm, title="Annotated Confusion Matrix", save_path=None):
    """
    Same as plot_confusion_matrix, but annotates each cell with a numeric count.
    
    Args:
        cm (ndarray): Confusion matrix (shape: (num_classes, num_classes)).
        title (str): Title of the plot.
        save_path (str, optional): If provided, file path to save the figure.
    """
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted Digit")
    plt.ylabel("True Digit")

    # Annotate each cell
    max_val = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(
                j, i, val,
                horizontalalignment="center",
                color="white" if val > (max_val / 2) else "black"
            )

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_history(train_values, val_values=None, title="Training History", 
                 ylabel="Error", xlabel="Epoch", save_path=None):
    """
    Plots a training (and optional validation) history over epochs.
    
    Args:
        train_values (list or ndarray): y-values for training set per epoch.
        val_values (list or ndarray, optional): y-values for validation set per epoch.
        title (str): Plot title.
        ylabel (str): Y-axis label.
        xlabel (str): X-axis label.
        save_path (str, optional): If provided, file path to save the figure.
    """
    epochs = range(1, len(train_values)+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_values, label="Train")
    if val_values is not None:
        plt.plot(epochs, val_values, label="Validation")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_class_metrics(values, metric_name="Metric", classes=None, save_path=None):
    """
    Plots a simple bar chart of a per-class metric (e.g. TPR, TNR, precision).
    
    Args:
        values (list or ndarray): Per-class metric values (one for each class).
        metric_name (str): Label for Y-axis, e.g. "TPR" or "Precision".
        classes (list, optional): Labels for x-axis. 
                                  If None, uses indices 0..len(values)-1.
        save_path (str, optional): If provided, file path to save the figure.
    """
    if classes is None:
        classes = list(range(len(values)))
    
    plt.figure(figsize=(10,5))
    plt.bar(classes, values, color='blue', alpha=0.7)
    plt.xlabel("Class")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} by Class")
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
