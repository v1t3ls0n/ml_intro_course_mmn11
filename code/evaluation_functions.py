# Evaluation Functions
# Imports and setup
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from custom_confusion_matrix import custom_confusion_matrix
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
    print("Prebuilt Confusion Matrix:\n", prebuilt_cm)


    # Custom confusion matrix implementation
    # custom_cm = custom_confusion_matrix(y, y_pred, num_classes=model.num_classes)
    # print("Custom Confusion Matrix:\n", custom_cm)



    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
    # Compute sensitivity for each class: TPR = TP / (TP + FN)
    for cls in range(model.num_classes):
        TP = prebuilt_cm[cls, cls]
        FN = np.sum(prebuilt_cm[cls, :]) - TP
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        print(f"Sensitivity for digit {cls}: {tpr:.2f}")
    
    return prebuilt_cm, accuracy

def plot_confusion_matrix(cm):
    """
    Plots the confusion matrix as a heatmap.
    """
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted Digit")
    plt.ylabel("True Digit")
    plt.show()
