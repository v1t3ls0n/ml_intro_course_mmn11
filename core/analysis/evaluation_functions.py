import numpy as np
from sklearn.metrics import confusion_matrix
from core.logger.config import logger
from core.analysis.custom_confusion_matrix import custom_confusion_matrix
from core.analysis.plotting import plot_confusion_matrix_annotated

def evaluate_model(model, X, y, classes, plot_dir):
    """
    Evaluates the trained model on data X with true labels y.
    Computes and logs:
      - Confusion matrices (built-in & custom).
      - Overall accuracy.
      - Sensitivity (TPR) for each class.
    Saves confusion matrix plot (annotated).
    
    Args:
        model: Trained classification model.
        X (ndarray): Data samples.
        y (ndarray): True labels.
        classes (list): List of class labels (e.g., [0,...,9] for MNIST).
        plot_dir (str): Directory to save confusion matrix plots.
        
    Returns:
        cm (np.ndarray): Built-in confusion matrix (scikit-learn).
        accuracy (float): Overall accuracy on X.
        sensitivity (list): List of sensitivity (TPR) per class.
    """
    y_pred = model.predict(X)

    # Built-in confusion matrix
    cm_builtin = confusion_matrix(y, y_pred)
    print("Built-in Confusion Matrix:\n{}".format(cm_builtin))

    # Custom confusion matrix
    custom_cm = custom_confusion_matrix(y, y_pred, num_classes=len(classes))
    print("Custom Confusion Matrix:\n{}".format(custom_cm))

    # Accuracy
    accuracy = np.trace(cm_builtin) / np.sum(cm_builtin)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

    # Sensitivity per class (TPR)
    sensitivity = []
    for cls in range(len(classes)):
        TP = cm_builtin[cls, cls]
        FN = np.sum(cm_builtin[cls, :]) - TP
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        sensitivity.append(tpr)
        print(f"Sensitivity (TPR) for digit '{classes[cls]}': {tpr:.2f}")

    # Plot annotated confusion matrix
    plot_confusion_matrix_annotated(
        cm_builtin, 
        classes=classes, 
        title="Annotated Confusion Matrix", 
        save_path=f"{plot_dir}/confusion_matrix_annotated.png"
    )
    # Use stored training runtime from the model
    runtime = model.training_runtime  
    return cm_builtin, accuracy, sensitivity, runtime
