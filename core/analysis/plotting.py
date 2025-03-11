import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix_annotated(cm, classes, title="Annotated Confusion Matrix", save_path=None, method="PLA-Default", max_iter=1000):
    """
    Plots a confusion matrix with numeric annotations in each cell.

    Args:
        cm (ndarray): Confusion matrix (num_classes x num_classes).
        classes (list): List of class labels (e.g., [0..9] for MNIST).
        title (str): Title for the plot.
        save_path (str): If provided, file path to save the figure.
        method (str): Indicating if it's "PLA-Clean" or "PLA-Pocket".
        max_iter (int): Maximum number of iterations used in training.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    
    # Adding the method and max_iter to the title
    full_title = f"{title}\n({method}, Max Iterations: {max_iter})"
    plt.title(full_title)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_error_curves(train_curve, test_curve, title, save_path=None, method="PLA-Clean", max_iter=1000):
    """
    Plots error curves for training and testing errors over iterations.

    Args:
        train_curve (list): List of training errors over iterations.
        test_curve (list): List of testing errors over iterations.
        title (str): Title for the plot.
        save_path (str): If provided, file path to save the figure.
        method (str): Indicating if it's "PLA-Clean" or "PLA-Pocket".
        max_iter (int): Maximum number of iterations used in training.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_curve, label='Training Error', marker='o')
    plt.plot(test_curve, label='Testing Error', marker='x')
    plt.xlabel('Iterations')
    plt.ylabel('Average Misclassification Error')
    
    # Adding the method and max_iter to the title
    full_title = f"{title}\n({method}, Max Iterations: {max_iter})"
    plt.title(full_title)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_accuracy_vs_max_iter(max_iter_values, accuracies_clean, accuracies_pocket, save_path=None):
    """
    Plots accuracy as a function of max_iter for both Clean and Pocket PLA.

    Args:
        max_iter_values (list): List of max_iter values.
        accuracies_clean (list): Corresponding accuracies for Clean PLA.
        accuracies_pocket (list): Corresponding accuracies for Pocket PLA.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(max_iter_values, accuracies_clean, marker='o', linestyle='-', label='Clean PLA')
    plt.plot(max_iter_values, accuracies_pocket, marker='s', linestyle='-', label='Pocket PLA')

    plt.xlabel("max_iter")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. max_iter")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_runtime_vs_max_iter(max_iter_values, runtimes_clean, runtimes_pocket, save_path=None):
    """
    Plots runtime as a function of max_iter for both Clean and Pocket PLA.

    Args:
        max_iter_values (list): List of max_iter values.
        runtimes_clean (list): Corresponding runtimes for Clean PLA.
        runtimes_pocket (list): Corresponding runtimes for Pocket PLA.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(max_iter_values, runtimes_clean, marker='o', linestyle='-', label='Clean PLA', color='blue')
    plt.plot(max_iter_values, runtimes_pocket, marker='s', linestyle='-', label='Pocket PLA', color='red')

    plt.xlabel("max_iter")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs. max_iter")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()
