import numpy as np
from core.logger.config import logger

class MultiClassPerceptron:
    def __init__(self, num_classes=10, max_iter=1000, use_pocket=True):
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.use_pocket = use_pocket
        self.weights = np.zeros((num_classes, 785))
        self.loss_history = {i: {"train": [], "val": [], "test": []} for i in range(num_classes)}
        self.converged_iterations = {}

    def fit(self, X, y, X_val=None, y_val=None, X_test=None, y_test=None):
        for cls in range(self.num_classes):
            binary_labels = np.where(y == cls, 1, -1)
            logger.info(f"Training classifier for class {cls}...")
            
            # pass y_val, y_test as val_labels, test_labels
            w, train_losses, val_losses, test_losses, iteration_count = self._train_binary(
                X, 
                binary_labels=binary_labels, 
                cls_idx=cls,
                X_val=X_val, 
                val_labels=y_val,
                X_test=X_test,
                test_labels=y_test
            )

            self.weights[cls] = w
            self.loss_history[cls]["train"] = train_losses
            self.loss_history[cls]["val"] = val_losses
            self.loss_history[cls]["test"] = test_losses
        
    def _train_binary(self, X, binary_labels, cls_idx, X_val=None, val_labels=None, X_test=None, test_labels=None):
        w = np.zeros(X.shape[1])
        best_w = w.copy()
        best_err = self._compute_error(X, binary_labels, w)
        
        train_losses, val_losses, test_losses = [], [], []
        for iteration in range(self.max_iter):
            preds = np.sign(X @ w)
            preds[preds == 0] = -1
            misclassified = preds != binary_labels
            
            # Update weights
            update = np.sum(X[misclassified] * binary_labels[misclassified][:, None], axis=0)
            w += update
            
            # Track errors
            current_err = self._compute_error(X, binary_labels, w)
            train_losses.append(current_err)

            if X_val is not None:
                val_err = self._compute_error(X_val, val_labels, w)
                val_losses.append(val_err)

            if X_test is not None:
                test_err = self._compute_error(X_test, test_labels, w)
                test_losses.append(test_err)

            # Pocket algorithm
            if self.use_pocket and current_err < best_err:
                best_w = w.copy()
                best_err = current_err

            # Early stopping if no misclassification
            if current_err == 0:
                logger.info(f"Classifier {cls_idx} converged after {len(train_losses)} iterations.")
                break

        # Choose best weights for Pocket PLA
        final_w = best_w if self.use_pocket else w
        iteration_count = len(train_losses)
        return best_w, train_losses, val_losses, test_losses, iteration_count
    
    def _compute_error(self, X, labels, w):
        preds = np.sign(np.dot(X, w))
        preds[preds == 0] = -1
        return np.sum(preds != labels)

    def predict(self, X):
        scores = np.dot(X, self.weights.T)
        return np.argmax(scores, axis=1)
