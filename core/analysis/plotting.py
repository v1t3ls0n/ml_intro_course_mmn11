import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix_annotated(cm, classes, title="Annotated Confusion Matrix", save_path=None):
    """
    Plots a confusion matrix with numeric annotations in each cell.

    Args:
        cm (ndarray): Confusion matrix (num_classes x num_classes).
        classes (list): List of class labels (e.g., [0..9] for MNIST).
        title (str): Title for the plot.
        save_path (str): If provided, file path to save the figure.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_error_curves(train_curve, test_curve, title, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_curve, label='Training Error', marker='o')
    plt.plot(test_curve, label='Testing Error', marker='x')
    plt.xlabel('Iterations')
    plt.ylabel('Average Misclassification Error')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
