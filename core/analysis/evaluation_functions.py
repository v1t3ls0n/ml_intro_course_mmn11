import numpy as np
from sklearn.metrics import confusion_matrix
from core.logger.config import logger

import sys
if sys.platform == 'win32':
    from tqdm import tqdm
else:
  from tqdm.notebook import tqdm

def evaluate_model(model, X, y, classes, plot_dir=None, train_curve=None, test_curve=None, show_plots=True, model_name=None):
    """
    Evaluates the trained model on data X with true labels y.
    Computes and logs:
      - Confusion matrix.
      - Overall accuracy.
      - Sensitivity (TPR) for each class.
      - Selectivity (Specificity) for each class.
    Optionally computes training/test loss curves, but does not plot any results.
    
    Args:
        model: Trained classification model.
        X (ndarray): Data samples.
        y (ndarray): True labels.
        classes (list): List of class labels.
        plot_dir (str): Directory to save any plots (not used here).
        train_curve (list, optional): Training loss values per iteration.
        test_curve (list, optional): Test loss values per iteration.
        show_plots (bool): Whether to display plots interactively (not used here).
        
    Returns:
        cm (np.ndarray): Confusion matrix.
        accuracy (float): Overall accuracy.
        sensitivity (list): Sensitivity (TPR) per class.
        selectivity (list): Selectivity (Specificity) per class.
        runtime: Training runtime.
        additional_info (dict): Dictionary containing extra info for plotting, e.g., 
                                 training and test curves, model name, and max_iter.
    """
    # Predict
    y_pred = model.predict(X)

    # Compute confusion matrix
    cm_builtin = confusion_matrix(y, y_pred)
    logger.info("Built-in Confusion Matrix:\n{}".format(cm_builtin))

    # Overall accuracy
    accuracy = np.trace(cm_builtin) / np.sum(cm_builtin)
    logger.info(f"Overall Accuracy: {accuracy * 100:.2f}%")

    # Compute per-class Sensitivity (TPR) and Selectivity (TNR)
    sensitivity = []
    selectivity = []
    for cls in tqdm(range(len(classes))):
        TP = cm_builtin[cls, cls]
        FN = np.sum(cm_builtin[cls, :]) - TP
        FP = np.sum(cm_builtin[:, cls]) - TP
        TN = np.sum(cm_builtin) - (TP + FN + FP)

        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        sensitivity.append(tpr)
        logger.info(f"Sensitivity (TPR) for class '{classes[cls]}': {tpr:.2f}")

        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        selectivity.append(specificity)
        logger.info(f"Selectivity (TNR) for class '{classes[cls]}': {specificity:.2f}")

    # Determine model name for plotting purposes
    model_name = getattr(model, "model_name", None)
    if model_name is not None:
        plot_method = model_name
    else:
        use_pocket = getattr(model, "use_pocket", False)
        plot_method = "PLA-Pocket" if use_pocket else "PLA-Clean"

    # Get max_iter if available
    max_iter = getattr(model, "max_iter", None)
    if max_iter is None:
        max_iter = 0

    additional_info = {
        "train_curve": train_curve,
        "test_curve": test_curve,
        "plot_method": plot_method,
        "max_iter": max_iter,
    }
    
    runtime = getattr(model, "training_runtime", None)
    
    return cm_builtin, accuracy, sensitivity, selectivity, runtime, additional_info



# Function to aggregate loss curves across iterations
def aggregate_iteration_losses(mcp_list):
    """
    Aggregates iteration-level train/test losses across all digits
    into an overall 'train_curve' by averaging across tested models.
    """
    num_classes = mcp_list[0].num_classes  # Assume all models have the same num_classes

    # Determine the maximum number of iterations across all models
    max_len = max(max(len(mcp.loss_history[cls_idx]["train"]) for cls_idx in range(num_classes)) for mcp in mcp_list)

    all_train_curves = []

    for mcp in tqdm(mcp_list):
        all_train = []
        for cls_idx in tqdm(range(num_classes)):
            t_arr = mcp.loss_history[cls_idx]["train"][:]

            # If classifier converged early, pad with last value
            if len(t_arr) < max_len:
                t_arr += [t_arr[-1]] * (max_len - len(t_arr))

            all_train.append(t_arr)

        # Convert to NumPy array and compute mean curve
        all_train = np.array(all_train)
        train_curve = np.mean(all_train, axis=0)

        all_train_curves.append(train_curve)

    # Convert all train curves into a uniform NumPy array
    all_train_curves = np.array(all_train_curves)

    return np.mean(all_train_curves, axis=0)  # Final averaged curve
