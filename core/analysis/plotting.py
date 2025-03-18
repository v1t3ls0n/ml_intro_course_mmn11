import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix_annotated(cm, classes, title="Annotated Confusion Matrix",
                                    save_path=None, method="ML Model Name", max_iter=1000):
    """
    Plots a confusion matrix with numeric annotations in each cell.
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
    Plots two curves (e.g. training vs. testing error) over iterations.
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


##############################################################################
# =========== Old iteration-based comparison functions (unchanged) ===========
##############################################################################

def plot_accuracy_vs_max_iter(
    max_iter_values,
    accuracies_clean,
    accuracies_pocket=None,
    accuracies_softmax=None,
    save_path=None
):
    """
    Plot accuracy vs. max_iter for up to three models: Clean, Pocket, Softmax.
    Skips lines if set to None.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(max_iter_values, accuracies_clean, marker='o', label='Clean PLA')
    if accuracies_pocket is not None:
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
    runtimes_pocket=None,
    runtimes_softmax=None,
    save_path=None
):
    """
    Plot runtime vs. max_iter for up to three models: Clean, Pocket, Softmax.
    Skips lines if set to None.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(max_iter_values, runtimes_clean, marker='o', label='Clean PLA')
    if runtimes_pocket is not None:
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
    Old iteration-based summary (Accuracy, TPR, TNR, Runtime vs. max_iter).
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 16))
    if accuracies_clean is not None:
        axes[0].plot(max_iter_values, np.array(accuracies_clean)*100, marker='o', label='Clean PLA')
    if accuracies_pocket is not None:
        axes[0].plot(max_iter_values, np.array(accuracies_pocket)*100, marker='s', label='Pocket PLA')
    if accuracies_softmax is not None:
        axes[0].plot(max_iter_values, np.array(accuracies_softmax)*100, marker='^', label='Softmax')
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Model Accuracy Comparison")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    if sensitivities_clean is not None:
        axes[1].plot(max_iter_values, np.array(sensitivities_clean)*100, marker='o', label='Clean PLA')
    if sensitivities_pocket is not None:
        axes[1].plot(max_iter_values, np.array(sensitivities_pocket)*100, marker='s', label='Pocket PLA')
    if sensitivities_softmax is not None:
        axes[1].plot(max_iter_values, np.array(sensitivities_softmax)*100, marker='^', label='Softmax')
    axes[1].set_ylabel("Sensitivity (TPR) (%)")
    axes[1].set_title("Model Sensitivity (TPR) Comparison")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()

    if selectivities_clean is not None:
        axes[2].plot(max_iter_values, np.array(selectivities_clean)*100, marker='o', label='Clean PLA')
    if selectivities_pocket is not None:
        axes[2].plot(max_iter_values, np.array(selectivities_pocket)*100, marker='s', label='Pocket PLA')
    if selectivities_softmax is not None:
        axes[2].plot(max_iter_values, np.array(selectivities_softmax)*100, marker='^', label='Softmax')
    axes[2].set_ylabel("Selectivity (TNR) (%)")
    axes[2].set_title("Model Selectivity (TNR) Comparison")
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].legend()

    if runtimes_clean is not None:
        axes[3].plot(max_iter_values, runtimes_clean, marker='o', label='Clean PLA')
    if runtimes_pocket is not None:
        axes[3].plot(max_iter_values, runtimes_pocket, marker='s', label='Pocket PLA')
    if runtimes_softmax is not None:
        axes[3].plot(max_iter_values, runtimes_softmax, marker='^', label='Softmax')
    axes[3].set_ylabel("Runtime (s)")
    axes[3].set_title("Model Runtime Comparison")
    axes[3].grid(True, linestyle='--', alpha=0.7)
    axes[3].legend()

    for ax in axes:
        ax.set_xlabel("Max Iterations")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


##############################################################################
# =========== New runtime-based comparison functions start here ==============
##############################################################################

def plot_accuracy_vs_runtime(
    runtimes_clean,
    accuracies_clean,
    runtimes_pocket=None,
    accuracies_pocket=None,
    runtimes_softmax=None,
    accuracies_softmax=None,
    title="Accuracy vs. Runtime (3-model)",
    save_path=None
):
    """
    Plots up to three lines of Accuracy vs. Runtime:
      - Clean PLA
      - Pocket PLA
      - Softmax
    If pocket or softmax data is None, skip that line.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(runtimes_clean, np.array(accuracies_clean)*100, marker='o', label="Clean PLA")
    if (runtimes_pocket is not None) and (accuracies_pocket is not None):
        plt.plot(runtimes_pocket, np.array(accuracies_pocket)*100, marker='s', label="Pocket PLA")
    if (runtimes_softmax is not None) and (accuracies_softmax is not None):
        plt.plot(runtimes_softmax, np.array(accuracies_softmax)*100, marker='^', label="Softmax")
    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_performance_summary_extended_by_runtime(
    runtimes_clean, accuracies_clean, sensitivities_clean, selectivities_clean,
    runtimes_pocket=None, accuracies_pocket=None, sensitivities_pocket=None, selectivities_pocket=None,
    runtimes_softmax=None, accuracies_softmax=None, sensitivities_softmax=None, selectivities_softmax=None,
    title="Performance vs. Runtime (3-model)",
    save_path=None
):
    """
    Plots a 3-panel summary of performance vs. runtime for up to three models:
      1) Accuracy vs. Runtime
      2) Sensitivity (TPR) vs. Runtime
      3) Selectivity (TNR) vs. Runtime
    for Clean, Pocket, Softmax. Skip lines if any is None.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    axes[0].plot(runtimes_clean,   np.array(accuracies_clean)*100, marker='o', label='Clean PLA')
    if (runtimes_pocket is not None) and (accuracies_pocket is not None):
        axes[0].plot(runtimes_pocket,  np.array(accuracies_pocket)*100, marker='s', label='Pocket PLA')
    if (runtimes_softmax is not None) and (accuracies_softmax is not None):
        axes[0].plot(runtimes_softmax, np.array(accuracies_softmax)*100, marker='^', label='Softmax')
    axes[0].set_xlabel("Runtime (seconds)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Accuracy vs. Runtime")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    axes[1].plot(runtimes_clean,   np.array(sensitivities_clean)*100, marker='o', label='Clean PLA')
    if (runtimes_pocket is not None) and (sensitivities_pocket is not None):
        axes[1].plot(runtimes_pocket,  np.array(sensitivities_pocket)*100, marker='s', label='Pocket PLA')
    if (runtimes_softmax is not None) and (sensitivities_softmax is not None):
        axes[1].plot(runtimes_softmax, np.array(sensitivities_softmax)*100, marker='^', label='Softmax')
    axes[1].set_xlabel("Runtime (seconds)")
    axes[1].set_ylabel("Sensitivity (TPR) (%)")
    axes[1].set_title("Sensitivity vs. Runtime")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    axes[2].plot(runtimes_clean,   np.array(selectivities_clean)*100, marker='o', label='Clean PLA')
    if (runtimes_pocket is not None) and (selectivities_pocket is not None):
        axes[2].plot(runtimes_pocket,  np.array(selectivities_pocket)*100, marker='s', label='Pocket PLA')
    if (runtimes_softmax is not None) and (selectivities_softmax is not None):
        axes[2].plot(runtimes_softmax, np.array(selectivities_softmax)*100, marker='^', label='Softmax')
    axes[2].set_xlabel("Runtime (seconds)")
    axes[2].set_ylabel("Selectivity (TNR) (%)")
    axes[2].set_title("Selectivity vs. Runtime")
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_performance_summary_4models_by_runtime(
    runtimes_clean, accuracies_clean, sensitivities_clean, selectivities_clean,
    runtimes_pocket, accuracies_pocket, sensitivities_pocket, selectivities_pocket,
    runtimes_softmax, accuracies_softmax, sensitivities_softmax, selectivities_softmax,
    runtimes_linear, accuracies_linear, sensitivities_linear, selectivities_linear,
    title="Performance vs. Runtime (4-model)",
    save_path=None
):
    """
    Plots 3 subplots (Accuracy, TPR, TNR) vs. runtime for four models:
      - Clean PLA
      - Pocket PLA
      - Softmax
      - Linear
    """
    fig, axes = plt.subplots(3, 1, figsize=(9, 14))
    axes[0].plot(runtimes_clean,   np.array(accuracies_clean)*100,    marker='o', label='Clean PLA')
    axes[0].plot(runtimes_pocket,  np.array(accuracies_pocket)*100,   marker='s', label='Pocket PLA')
    axes[0].plot(runtimes_softmax, np.array(accuracies_softmax)*100,  marker='^', label='Softmax')
    axes[0].plot(runtimes_linear,  np.array(accuracies_linear)*100,   marker='d', label='Linear')
    axes[0].set_xlabel("Runtime (seconds)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Accuracy vs. Runtime")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)

    axes[1].plot(runtimes_clean,   np.array(sensitivities_clean)*100,    marker='o', label='Clean PLA')
    axes[1].plot(runtimes_pocket,  np.array(sensitivities_pocket)*100,   marker='s', label='Pocket PLA')
    axes[1].plot(runtimes_softmax, np.array(sensitivities_softmax)*100,  marker='^', label='Softmax')
    axes[1].plot(runtimes_linear,  np.array(sensitivities_linear)*100,   marker='d', label='Linear')
    axes[1].set_xlabel("Runtime (seconds)")
    axes[1].set_ylabel("Sensitivity (TPR) (%)")
    axes[1].set_title("Sensitivity vs. Runtime")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)

    axes[2].plot(runtimes_clean,   np.array(selectivities_clean)*100,    marker='o', label='Clean PLA')
    axes[2].plot(runtimes_pocket,  np.array(selectivities_pocket)*100,   marker='s', label='Pocket PLA')
    axes[2].plot(runtimes_softmax, np.array(selectivities_softmax)*100,  marker='^', label='Softmax')
    axes[2].plot(runtimes_linear,  np.array(selectivities_linear)*100,   marker='d', label='Linear')
    axes[2].set_xlabel("Runtime (seconds)")
    axes[2].set_ylabel("Selectivity (TNR) (%)")
    axes[2].set_title("Selectivity vs. Runtime")
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_accuracy_vs_runtime_4models(
    rt_clean, acc_clean,
    rt_pocket, acc_pocket,
    rt_softmax, acc_softmax,
    rt_linear, acc_linear,
    title="Accuracy vs. Runtime (4 models)",
    save_path=None
):
    """
    Plots a single chart of Accuracy vs. Runtime for four models:
      Clean, Pocket, Softmax, Linear
    """
    plt.figure(figsize=(8, 5))
    plt.plot(rt_clean,   np.array(acc_clean)*100,   marker='o', label='Clean PLA')
    plt.plot(rt_pocket,  np.array(acc_pocket)*100,  marker='s', label='Pocket PLA')
    plt.plot(rt_softmax, np.array(acc_softmax)*100, marker='^', label='Softmax')
    plt.plot(rt_linear,  np.array(acc_linear)*100,  marker='d', label='Linear')
    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_train_curves_three_models(
    clean_train_curve=None,
    pocket_train_curve=None,
    softmax_train_curve=None,
    title="Aggregated Training Curves",
    max_iter=0,
    save_path=None
):
    """
    Plots up to three aggregated training-loss curves (Clean, Pocket, Softmax).
    If any curve is None, it is skipped.
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


def plot_accuracy_vs_max_iter_4models(
    max_iter_values,
    acc_clean,
    acc_pocket,
    acc_softmax,
    acc_linear,
    save_path=None
):
    """
    Plots accuracy vs. max_iter for four models: Clean, Pocket, Softmax, Linear.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(max_iter_values, acc_clean,   marker='o', label='Clean PLA')
    plt.plot(max_iter_values, acc_pocket,  marker='s', label='Pocket PLA')
    plt.plot(max_iter_values, acc_softmax, marker='^', label='Softmax')
    plt.plot(max_iter_values, acc_linear,  marker='d', label='Linear')
    plt.xlabel("max_iter")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. max_iter (4 models)")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_runtime_vs_max_iter_4models(
    max_iter_values,
    rt_clean,
    rt_pocket,
    rt_softmax,
    rt_linear,
    save_path=None
):
    """
    Plots runtime vs. max_iter for four models: Clean, Pocket, Softmax, Linear.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(max_iter_values, rt_clean,   marker='o', label='Clean PLA')
    plt.plot(max_iter_values, rt_pocket,  marker='s', label='Pocket PLA')
    plt.plot(max_iter_values, rt_softmax, marker='^', label='Softmax')
    plt.plot(max_iter_values, rt_linear,  marker='d', label='Linear')
    plt.xlabel("max_iter")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs. max_iter (4 models)")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
