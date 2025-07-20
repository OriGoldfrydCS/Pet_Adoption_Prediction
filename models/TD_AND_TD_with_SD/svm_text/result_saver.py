"""
Utility for saving all outputs from a scikit-learn model run

Includes:
- Saving model and its hyperparameters
- Classification report, accuracy, F1, and log loss
- Confusion matrix and performance plots (ROC, PR curves)
"""

import os
import joblib
import json
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import log_loss

def save_run_outputs(model, X, y_true, y_pred, name, base_output_dir, use_timestamp_subfolder=True):
    """
    Saves the full evaluation outputs for a trained scikit-learn model:
    - Model weights and hyperparameters
    - Accuracy, F1, and log loss (if applicable)
    - Classification report
    - Class distribution and confusion matrix
    - ROC and Precision-Recall curves

    Args:
        - model: Trained scikit-learn model.
        - X: Input features used for prediction.
        - y_true: True labels.
        - y_pred: Predicted labels.
        - use_timestamp_subfolder (bool): Whether to create a timestamped subdirectory.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, run_id) if use_timestamp_subfolder else base_output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    joblib.dump(model, os.path.join(output_dir, "model.joblib"))

    # Save hyperparameters
    if hasattr(model, "get_params"):
        params = model.get_params()
        with open(os.path.join(output_dir, "model_params.txt"), "w") as f:
            json.dump(params, f, indent=4)

    # Save classification report
    with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    # Save accuracy and F1
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nF1 (macro): {f1:.4f}\n")

    # Save class distributions
    pd.Series(y_true).value_counts().to_csv(os.path.join(output_dir, f"{name}_true_distribution.txt"))
    pd.Series(y_pred).value_counts().to_csv(os.path.join(output_dir, f"{name}_pred_distribution.txt"))

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name.capitalize()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"))
    plt.close()

    # Log loss if probabilities are available
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

    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")
        if logloss is not None:
            f.write(f"Log Loss: {logloss:.4f}\n")

    # Probability-based plots
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)
            pd.DataFrame(y_proba).to_csv(os.path.join(output_dir, f"{name}_probabilities.csv"), index=False)

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
            plt.savefig(os.path.join(output_dir, f"{name}_roc_curve.png"))
            plt.close()

            # Precision–Recall Curve
            for class_id in np.unique(y_true):
                precision, recall, _ = precision_recall_curve(y_true == class_id, y_proba[:, class_id])
                plt.plot(recall, precision, label=f"Class {class_id}")

            plt.title(f"{name.capitalize()} Precision–Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{name}_pr_curve.png"))
            plt.close()

        except Exception as e:
            print(f"Skipping probability-based plots: {e}")

    print(f"Results saved in: {output_dir}")
