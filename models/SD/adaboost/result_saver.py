"""
Utility module for saving AdaBoost model outputs after evaluation.

This script:
- Saves the trained model.
- Writes a classification report, accuracy, macro-F1, and optionally log loss.
- Saves class distribution counts for true and predicted labels.
- Generates and saves a confusion matrix as an image.
- Saves predicted probabilities.
- Plots ROC and Precision–Recall curves per class.
- Saves model hyperparameters to file.
"""

import os
import joblib
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc, log_loss
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def save_run_outputs(model, X, y_true, y_pred, name, base_output_dir):
    """
    Saves all relevant outputs of a model evaluation run.

    Parameters:
    - model: Trained classifier.
    - X: Features used for prediction.
    - y_true: Ground truth labels.
    - y_pred: Model predictions.
    """

    os.makedirs(base_output_dir, exist_ok=True)

    # Save model only once (assumes called first for validation)
    model_path = os.path.join(base_output_dir, "model.joblib")
    if not os.path.exists(model_path):
        joblib.dump(model, model_path)

    # Save classification report
    with open(os.path.join(base_output_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    # Accuracy and F1
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    with open(os.path.join(base_output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nF1 (macro): {f1:.4f}\n")

    # Save class distributions (for debugging/imbalance analysis)
    pd.Series(y_true).value_counts().to_csv(os.path.join(base_output_dir, f"{name}_true_distribution.txt"))
    pd.Series(y_pred).value_counts().to_csv(os.path.join(base_output_dir, f"{name}_pred_distribution.txt"))

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name.capitalize()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, f"{name}_confusion_matrix.png"))
    plt.close()

    # Optional: Log loss (only if model has predict_proba and all classes match)
    logloss = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)
            if y_proba.shape[1] == len(np.unique(y_true)):
                logloss = log_loss(y_true, y_proba)
            else:
                print("Mismatch in number of classes for log loss. Skipping.")
        except Exception as e:
            print(f"Error computing log loss: {e}")

    # Rewrite metrics with log loss if available
    with open(os.path.join(base_output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")
        if logloss is not None:
            f.write(f"Log Loss: {logloss:.4f}\n")

    # Probabilities + ROC/PR curves
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)

            # Save raw probabilities
            pd.DataFrame(y_proba).to_csv(os.path.join(base_output_dir, f"{name}_probabilities.csv"), index=False)

            # ROC Curve
            for class_id in np.unique(y_true):
                fpr, tpr, _ = roc_curve(y_true == class_id, y_proba[:, class_id])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"Class {class_id} (AUC={roc_auc:.2f})")

            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.title(f"{name.capitalize()} ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(base_output_dir, f"{name}_roc_curve.png"))
            plt.close()

            # PR Curve
            for class_id in np.unique(y_true):
                precision, recall, _ = precision_recall_curve(y_true == class_id, y_proba[:, class_id])
                plt.plot(recall, precision, label=f"Class {class_id}")

            plt.title(f"{name.capitalize()} Precision–Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(base_output_dir, f"{name}_pr_curve.png"))
            plt.close()

            # Save model hyperparameters
            with open(os.path.join(base_output_dir, "model_params.txt"), "w") as f:
                f.write(str(model.get_params()))

        except Exception as e:
            print(f"[Warning] Skipped probability-based plots due to: {e}")

    print(f"Results saved in: {base_output_dir}")
