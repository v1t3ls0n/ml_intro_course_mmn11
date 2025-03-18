import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix_annotated(cm, classes, title="Annotated Confusion Matrix",
                                    save_path=None, method="ML Model Name", max_iter=1000):
    """
    Plots a confusion matrix with numeric annotations in each cell.

    Args:
        cm (ndarray): Confusion matrix (num_classes x num_classes).
        classes (list): List of class labels (e.g., digits [0..9]).
        title (str): Title for the plot.
        save_path (str): If provided, file path to save the figure.
        method (str): Label for the method/model name (e.g. "PLA-Clean", "Pocket PLA", "Softmax").
        max_iter (int): Maximum number of iterations used in training (for display).
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    full_title = f"{title}\n({method}, Max Iterations: {max_iter})"
    plt.title(full_title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_error_curves(train_curve, test_curve, title, save_path=None,
                      method="ML Model Name", max_iter=1000):
    """
    Plots two curves (e.g. training error vs. testing error) over iterations.

    Args:
        train_curve (list): Training errors over iterations.
        test_curve (list): Testing errors over iterations.
        title (str): Plot title.
        save_path (str): If provided, file path to save the figure.
        method (str): Model name label for the plot.
        max_iter (int): Max iterations used in training, for display.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_curve, label='Training Error', marker='o')
    plt.plot(test_curve, label='Testing Error', marker='x')
    plt.xlabel('Iterations')
    plt.ylabel('Average Misclassification Error')
    full_title = f"{title}\n({method}, Max Iterations: {max_iter})"
    plt.title(full_title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_accuracy_vs_max_iter(
    max_iter_values,
    accuracies_clean,
    accuracies_pocket,
    accuracies_softmax=None,
    save_path=None
):
    """
    Plots accuracy vs. max_iter for up to three models:
      - Clean PLA
      - Pocket PLA
      - Softmax (optional)

    Args:
        max_iter_values (list): List of max_iter values.
        accuracies_clean (list): Accuracy for Clean PLA at each max_iter.
        accuracies_pocket (list): Accuracy for Pocket PLA at each max_iter.
        accuracies_softmax (list, optional): Accuracy for Softmax at each max_iter.
        save_path (str, optional): If provided, path to save the figure.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(max_iter_values, accuracies_clean, marker='o', label='Clean PLA')
    plt.plot(max_iter_values, accuracies_pocket, marker='s', label='Pocket PLA')

    if accuracies_softmax is not None:
        plt.plot(max_iter_values, accuracies_softmax, marker='^', label='Softmax')

    plt.xlabel("max_iter")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. max_iter")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_runtime_vs_max_iter(
    max_iter_values,
    runtimes_clean,
    runtimes_pocket,
    runtimes_softmax=None,
    save_path=None
):
    """
    Plots runtime vs. max_iter for up to three models:
      - Clean PLA
      - Pocket PLA
      - Softmax (optional)

    Args:
        max_iter_values (list): List of max_iter values.
        runtimes_clean (list): Runtimes for Clean PLA.
        runtimes_pocket (list): Runtimes for Pocket PLA.
        runtimes_softmax (list, optional): Runtimes for Softmax, if available.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(max_iter_values, runtimes_clean, marker='o', label='Clean PLA')
    plt.plot(max_iter_values, runtimes_pocket, marker='s', label='Pocket PLA')

    if runtimes_softmax is not None:
        plt.plot(max_iter_values, runtimes_softmax, marker='^', label='Softmax')

    plt.xlabel("max_iter")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs. max_iter")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_performance_summary_extended(
    max_iter_values,
    accuracies_clean=None,   accuracies_pocket=None,   accuracies_softmax=None,
    sensitivities_clean=None, sensitivities_pocket=None, sensitivities_softmax=None,
    selectivities_clean=None, selectivities_pocket=None, selectivities_softmax=None,
    runtimes_clean=None,     runtimes_pocket=None,     runtimes_softmax=None,
    save_path=None
):
    """
    Plots a comprehensive summary of performance metrics across different max_iter values.
    Includes: Accuracy, Sensitivity (TPR), Selectivity (TNR), and Runtime.
    Supports up to three models: Clean PLA, Pocket PLA, and optional Softmax.

    Args:
        max_iter_values (list): List of max iteration values.
        accuracies_clean (list[float]): Accuracy for PLA Clean.
        accuracies_pocket (list[float]): Accuracy for PLA Pocket.
        accuracies_softmax (list[float], optional): Accuracy for Softmax.
        sensitivities_clean (list[float]): TPR for PLA Clean.
        sensitivities_pocket (list[float]): TPR for PLA Pocket.
        sensitivities_softmax (list[float], optional): TPR for Softmax.
        selectivities_clean (list[float]): TNR for PLA Clean.
        selectivities_pocket (list[float]): TNR for PLA Pocket.
        selectivities_softmax (list[float], optional): TNR for Softmax.
        runtimes_clean (list[float]): Runtime for PLA Clean.
        runtimes_pocket (list[float]): Runtime for PLA Pocket.
        runtimes_softmax (list[float], optional): Runtime for Softmax.
        save_path (str, optional): If provided, saves the figure to this path.
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 16))

    # 1) Accuracy
    axes[0].plot(max_iter_values, np.array(accuracies_clean) * 100, marker='o', label='Clean PLA')
    axes[0].plot(max_iter_values, np.array(accuracies_pocket) * 100, marker='s', label='Pocket PLA')
    if accuracies_softmax is not None:
        axes[0].plot(max_iter_values, np.array(accuracies_softmax) * 100, marker='^', label='Softmax')
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xlabel("Max Iterations")
    axes[0].set_title("Model Accuracy Comparison")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    # 2) Sensitivity (TPR)
    axes[1].plot(max_iter_values, np.array(sensitivities_clean) * 100, marker='o', label='Clean PLA')
    axes[1].plot(max_iter_values, np.array(sensitivities_pocket) * 100, marker='s', label='Pocket PLA')
    if sensitivities_softmax is not None:
        axes[1].plot(max_iter_values, np.array(sensitivities_softmax) * 100, marker='^', label='Softmax')
    axes[1].set_ylabel("Sensitivity (TPR) (%)")
    axes[1].set_xlabel("Max Iterations")
    axes[1].set_title("Model Sensitivity (TPR) Comparison")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()

    # 3) Selectivity (TNR)
    axes[2].plot(max_iter_values, np.array(selectivities_clean) * 100, marker='o', label='Clean PLA')
    axes[2].plot(max_iter_values, np.array(selectivities_pocket) * 100, marker='s', label='Pocket PLA')
    if selectivities_softmax is not None:
        axes[2].plot(max_iter_values, np.array(selectivities_softmax) * 100, marker='^', label='Softmax')
    axes[2].set_ylabel("Selectivity (TNR) (%)")
    axes[2].set_xlabel("Max Iterations")
    axes[2].set_title("Model Selectivity (TNR) Comparison")
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].legend()

    # 4) Runtime
    axes[3].plot(max_iter_values, runtimes_clean, marker='o', label='Clean PLA')
    axes[3].plot(max_iter_values, runtimes_pocket, marker='s', label='Pocket PLA')
    if runtimes_softmax is not None:
        axes[3].plot(max_iter_values, runtimes_softmax, marker='^', label='Softmax')
    axes[3].set_xlabel("Max Iterations")
    axes[3].set_ylabel("Runtime (s)")
    axes[3].set_title("Model Runtime Comparison")
    axes[3].grid(True, linestyle='--', alpha=0.7)
    axes[3].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


#
# ===== NEW FUNCTION: Plot Aggregated Training Curves for Up to Three Models =====
#
def plot_train_curves_three_models(
    clean_train_curve=None,
    pocket_train_curve=None,
    softmax_train_curve=None,
    title="Aggregated Training Curves",
    max_iter=0,
    save_path=None
):
    """
    Plots up to three training-loss curves (or error curves) side by side:
      - Clean PLA
      - Pocket PLA
      - Softmax

    Args:
        clean_train_curve (list or None): Aggregated train curve for Clean PLA
        pocket_train_curve (list or None): Aggregated train curve for Pocket PLA
        softmax_train_curve (list or None): Aggregated train curve for Softmax
        title (str): Title for the plot
        max_iter (int): Max iteration (for display in title)
        save_path (str, optional): Path to save figure if desired.
    """
    plt.figure(figsize=(8, 5))

    if clean_train_curve is not None:
        plt.plot(clean_train_curve, marker='o', label="Clean PLA")
    if pocket_train_curve is not None:
        plt.plot(pocket_train_curve, marker='s', label="Pocket PLA")
    if softmax_train_curve is not None:
        plt.plot(softmax_train_curve, marker='^', label="Softmax")

    plt.xlabel("Iterations")
    plt.ylabel("Average Training Loss / Error")
    full_title = f"{title}\n(Max Iter={max_iter})"
    plt.title(full_title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_metric_vs_learning_rate(learning_rates, metric_values, metric_name="Accuracy",
                                 use_log_scale=True, save_path=None):
    """
    Plots a given metric (e.g. accuracy, sensitivity, runtime) vs. a range of learning rates.

    Args:
        learning_rates (list): List of learning rates tested (e.g. [0.001, 0.01, 0.1]).
        metric_values (list): Metric values corresponding to each learning rate.
        metric_name (str): Name of the metric to plot on the y-axis (e.g. "Accuracy").
        use_log_scale (bool): Whether to plot the x-axis on log scale.
        save_path (str, optional): If provided, file path to save the figure.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(learning_rates, metric_values, marker='o', label=f"{metric_name} vs. LR")

    if use_log_scale:
        plt.xscale("log")

    plt.xlabel("Learning Rate")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs. Learning Rate")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
