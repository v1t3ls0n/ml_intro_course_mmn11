import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from core.logger.config import logger
# from core.analysis.custom_confusion_matrix import custom_confusion_matrix  # Uncomment if needed

def evaluate_model(model, X, y, classes, plot_dir, train_curve=None, test_curve=None, show_plots=True):
    """
    Evaluates the trained model on data X with true labels y.
    Computes and logs:
      - Confusion matrix (built-in, with annotations).
      - Overall accuracy.
      - Sensitivity (TPR) for each class.
      - Selectivity (Specificity) for each class.
    Also visualizes:
      - The annotated confusion matrix.
      - (Optionally) the training and test loss curves vs. iteration.
      
    Args:
        model: Trained classification model.
        X (ndarray): Data samples.
        y (ndarray): True labels.
        classes (list): List of class labels (e.g., [0,...,9] for MNIST).
        plot_dir (str): Directory to save the plots.
        train_curve (list, optional): Training loss values per iteration.
        test_curve (list, optional): Test loss values per iteration.
        show_plots (bool): Whether to display plots interactively.
        
    Returns:
        cm (np.ndarray): Built-in confusion matrix (scikit-learn).
        accuracy (float): Overall accuracy on X.
        sensitivity (list): List of sensitivity (TPR) per class.
        selectivity (list): List of selectivity (specificity) per class.
        runtime: Training runtime stored in the model.
    """
    # Get model predictions
    y_pred = model.predict(X)

    # Built-in confusion matrix
    cm_builtin = confusion_matrix(y, y_pred)
    logger.info("Built-in Confusion Matrix:\n{}".format(cm_builtin))

    # Accuracy
    accuracy = np.trace(cm_builtin) / np.sum(cm_builtin)
    logger.info(f"Overall Accuracy: {accuracy * 100:.2f}%")

    # Sensitivity (TPR) and Selectivity (Specificity) per class
    sensitivity = []
    selectivity = []
    for cls in range(len(classes)):
        TP = cm_builtin[cls, cls]
        FN = np.sum(cm_builtin[cls, :]) - TP
        FP = np.sum(cm_builtin[:, cls]) - TP
        TN = np.sum(cm_builtin) - (TP + FN + FP)
        
        # Sensitivity: True Positive Rate
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        sensitivity.append(tpr)
        logger.info(f"Sensitivity (TPR) for class '{classes[cls]}': {tpr:.2f}")

        # Selectivity: Specificity (True Negative Rate)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        selectivity.append(specificity)
        logger.info(f"Selectivity (Specificity) for class '{classes[cls]}': {specificity:.2f}")

    # Plot: Annotated Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_builtin, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plot_method = "PLA-Pocket" if getattr(model, "use_pocket", False) else "PLA-Clean"
    full_title = f"Annotated Confusion Matrix\n({plot_method}, Max Iterations: {model.max_iter})"
    plt.title(full_title)
    if plot_dir:
        cm_path = f"{plot_dir}/confusion_matrix_annotated.png"
        plt.savefig(cm_path, dpi=300)
        logger.info(f"Confusion matrix saved to {cm_path}")
    if show_plots:
        plt.show()
    plt.close()

    # Plot: Training and Test Loss Curves (if available)
    if train_curve is not None and test_curve is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(train_curve, label='Training Loss', marker='o')
        plt.plot(test_curve, label='Test Loss', marker='x')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title("Training and Test Loss vs. Iteration")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        if plot_dir:
            loss_path = f"{plot_dir}/loss_curve.png"
            plt.savefig(loss_path, dpi=300)
            logger.info(f"Loss curves saved to {loss_path}")
        if show_plots:
            plt.show()
        plt.close()

    # Use stored training runtime from the model
    runtime = getattr(model, "training_runtime", None)
    
    return cm_builtin, accuracy, sensitivity, selectivity, runtime
