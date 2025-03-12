import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from core.logger.config import logger

def evaluate_model(model, X, y, classes, plot_dir=None, train_curve=None, test_curve=None, show_plots=True, model_name=None):
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
        model: Trained classification model. Must have:
            - predict(X) method returning integer labels
            - training_runtime attribute (optional)
            - max_iter attribute (optional)
            - model_name attribute (optional) => e.g., "Softmax", "PLA-Clean", "PLA-Pocket"
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
    # Predict
    y_pred = model.predict(X)

    # Compute confusion matrix
    cm_builtin = confusion_matrix(y, y_pred)
    logger.info("Built-in Confusion Matrix:\n{}".format(cm_builtin))

    # Overall accuracy
    accuracy = np.trace(cm_builtin) / np.sum(cm_builtin)
    logger.info(f"Overall Accuracy: {accuracy * 100:.2f}%")

    # Per-class Sensitivity (TPR) and Selectivity (TNR)
    sensitivity = []
    selectivity = []
    for cls in range(len(classes)):
        TP = cm_builtin[cls, cls]
        FN = np.sum(cm_builtin[cls, :]) - TP
        FP = np.sum(cm_builtin[:, cls]) - TP
        TN = np.sum(cm_builtin) - (TP + FN + FP)

        # Sensitivity (TPR)
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        sensitivity.append(tpr)
        logger.info(f"Sensitivity (TPR) for class '{classes[cls]}': {tpr:.2f}")

        # Selectivity (TNR)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        selectivity.append(specificity)
        logger.info(f"Selectivity (TNR) for class '{classes[cls]}': {specificity:.2f}")

    # Determine model name (for plot titles)
    model_name = getattr(model, "model_name", None)
    if model_name is not None:
        plot_method = model_name
    else:
        # Fallback to original PLA logic
        use_pocket = getattr(model, "use_pocket", False)
        plot_method = "PLA-Pocket" if use_pocket else "PLA-Clean"

    # Get max_iter if it exists
    max_iter = getattr(model, "max_iter", None)
    if max_iter is None:
        max_iter = 0  # or just use 'Unknown'

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_builtin, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    full_title = f"Annotated Confusion Matrix\n({plot_method}, Max Iterations: {max_iter})"
    plt.title(full_title)
    if plot_dir:
        cm_path = f"{plot_dir}/confusion_matrix_annotated.png"
        plt.savefig(cm_path, dpi=300)
        logger.info(f"Confusion matrix saved to {cm_path}")
    if show_plots:
        plt.show()
    plt.close()

    # Plot optional training/test loss curves
    if train_curve is not None or test_curve is not None:
        plt.figure(figsize=(10, 6))
        if train_curve is not None:
            plt.plot(train_curve, label='Training Loss', marker='o')
        if test_curve is not None:
            plt.plot(test_curve, label='Test Loss', marker='x')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f"Training and Test Loss vs. Iteration ({plot_method})")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        if plot_dir:
            loss_path = f"{plot_dir}/loss_curve.png"
            plt.savefig(loss_path, dpi=300)
            logger.info(f"Loss curves saved to {loss_path}")
        if show_plots:
            plt.show()
        plt.close()

    # Training runtime
    runtime = getattr(model, "training_runtime", None)
    
    return cm_builtin, accuracy, sensitivity, selectivity, runtime
