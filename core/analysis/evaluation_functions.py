import numpy as np
from sklearn.metrics import confusion_matrix
from core.logger.config import logger

import sys
if sys.platform == 'win32':
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm

def evaluate_model(model, X, y, classes, plot_dir=None, train_curve=None,
                   test_curve=None, show_plots=True, model_name=None):
    """
    Evaluates the trained model on data X with true labels y.
    Computes and logs:
      - Confusion matrix
      - Overall accuracy
      - Sensitivity (TPR) per class
      - Selectivity (Specificity, TNR) per class

    Args:
        model: Trained classification model.
        X (ndarray): Data samples.
        y (ndarray): True labels.
        classes (list): List of class labels.
        plot_dir (str): Directory to save any plots (not used here).
        train_curve (list, optional): Training loss values per iteration.
        test_curve (list, optional): Test loss values per iteration.
        show_plots (bool): Whether to display plots interactively (not used here).
        model_name (str, optional): A string identifier for the model 
            (e.g. "Clean PLA", "Pocket PLA", "Softmax").
            If not provided, we attempt to deduce from the model’s attributes.

    Returns:
        cm (np.ndarray): Confusion matrix.
        accuracy (float): Overall accuracy.
        sensitivity (list): Sensitivity (TPR) per class.
        selectivity (list): Selectivity (TNR) per class.
        runtime (float or None): Training runtime if tracked in the model.
        additional_info (dict): Extra info for plotting (loss curves, model name,
            max_iter, learning_rate, etc.).
    """
    # ---------- 1) Predictions ----------
    y_pred = model.predict(X)

    # ---------- 2) Confusion Matrix ----------
    cm_builtin = confusion_matrix(y, y_pred)
    logger.info("Built-in Confusion Matrix:\n{}".format(cm_builtin))

    # ---------- 3) Overall Accuracy ----------
    accuracy = np.trace(cm_builtin) / np.sum(cm_builtin)
    logger.info(f"Overall Accuracy: {accuracy * 100:.2f}%")

    # ---------- 4) Class-wise TPR & TNR ----------
    sensitivity = []
    selectivity = []
    for cls in tqdm(range(len(classes)), desc="Evaluating class metrics"):
        TP = cm_builtin[cls, cls]
        FN = np.sum(cm_builtin[cls, :]) - TP
        FP = np.sum(cm_builtin[:, cls]) - TP
        TN = np.sum(cm_builtin) - (TP + FN + FP)

        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        tnr = TN / (TN + FP) if (TN + FP) > 0 else 0

        sensitivity.append(tpr)
        selectivity.append(tnr)

        logger.info(f"Class '{classes[cls]}': TPR={tpr:.2f}, TNR={tnr:.2f}")

    # ---------- 5) Determine Model Name for Plotting ----------
    if model_name is not None:
        plot_method = model_name
    else:
        if hasattr(model, "model_name"):
            plot_method = model.model_name
        else:
            # Try to detect from known attributes or class name
            use_pocket = getattr(model, "use_pocket", None)
            if use_pocket is True:
                plot_method = "Pocket PLA"
            elif use_pocket is False:
                plot_method = "Clean PLA"
            else:
                if 'SoftmaxRegression' in str(type(model)):
                    plot_method = "Softmax Regression"
                else:
                    plot_method = "GenericModel"

    # ---------- 6) Extract max_iter (if available) ----------
    max_iter = getattr(model, "max_iter", 0)

    # ---------- 7) Extract & Store Learning Rate (if available) ----------
    learning_rate = getattr(model, "learning_rate", None)

    # ---------- Additional Info ----------
    additional_info = {
        "train_curve": train_curve,
        "test_curve": test_curve,
        "plot_method": plot_method,
        "max_iter": max_iter,
        "learning_rate": learning_rate
    }

    # ---------- Attempt to retrieve training runtime ----------
    runtime = getattr(model, "training_runtime", None)

    return cm_builtin, accuracy, sensitivity, selectivity, runtime, additional_info


def aggregate_iteration_losses(mcp_list):
    """
    Aggregates iteration-level train/test losses for a list of
    MultiClassPerceptron models. Produces an overall 'train_curve'
    by averaging across all classes and all given perceptron models.

    NOTE: This function is tailored to the MultiClassPerceptron,
          where each class i has a list of losses in:
          mcp.loss_history[i]["train"].

    Args:
        mcp_list (list): List of MultiClassPerceptron objects.

    Returns:
        avg_curve (list or np.ndarray):
            The averaged training-loss curve across all provided models.
    """
    if not mcp_list:
        return []

    num_classes = mcp_list[0].num_classes

    # Determine the maximum number of recorded iterations
    max_len = 0
    for mcp in mcp_list:
        for cls_idx in range(num_classes):
            length = len(mcp.loss_history[cls_idx]["train"])
            if length > max_len:
                max_len = length

    # Collect the train curves from each model
    all_models_train_curves = []

    for mcp in tqdm(mcp_list, desc="Aggregating train losses across Perceptron models"):
        class_train_curves = []
        for cls_idx in range(num_classes):
            t_arr = mcp.loss_history[cls_idx]["train"][:]
            # If a class converged early, pad with the last value
            if len(t_arr) < max_len:
                t_arr += [t_arr[-1]] * (max_len - len(t_arr))
            class_train_curves.append(t_arr)

        class_train_curves = np.array(class_train_curves)
        # Average across classes for this particular model
        model_avg_curve = np.mean(class_train_curves, axis=0)
        all_models_train_curves.append(model_avg_curve)

    # Now average across all models
    all_models_train_curves = np.array(all_models_train_curves)
    avg_curve = np.mean(all_models_train_curves, axis=0)

    return avg_curve


def aggregate_iteration_losses_softmax(softmax_list):
    """
    Aggregates iteration-level training losses for a list of SoftmaxRegression models.
    Produces one overall train-loss curve by:
      1) Averaging across classes (within each model),
      2) Averaging across all the models in softmax_list.

    Args:
        softmax_list (list): List of SoftmaxRegression objects.

    Returns:
        avg_curve (list or np.ndarray):
            The averaged training-loss curve across the given models.
    """
    if not softmax_list:
        return []

    # All models are assumed to have the same self.num_classes
    num_classes = softmax_list[0].num_classes

    # 1) Find max length of iteration history across all models/classes
    max_len = 0
    for model in softmax_list:
        for cls_idx in range(num_classes):
            length = len(model.loss_history[cls_idx]["train"])
            if length > max_len:
                max_len = length

    # 2) For each model in softmax_list, gather & pad class losses
    all_models_train_curves = []
    for model in softmax_list:
        class_train_curves = []
        for cls_idx in range(num_classes):
            t_arr = model.loss_history[cls_idx]["train"][:]
            # pad if needed
            if len(t_arr) < max_len:
                t_arr += [t_arr[-1]] * (max_len - len(t_arr))
            class_train_curves.append(t_arr)

        # average across classes for this model
        class_train_curves = np.array(class_train_curves)
        model_avg_curve = np.mean(class_train_curves, axis=0)
        all_models_train_curves.append(model_avg_curve)

    # 3) Average across all models
    all_models_train_curves = np.array(all_models_train_curves)
    avg_curve = np.mean(all_models_train_curves, axis=0)

    return avg_curve
