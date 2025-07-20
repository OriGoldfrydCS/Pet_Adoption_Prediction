"""
Utility for saving all outputs from a model run - Logistic Regression with BERT embeddings.

Includes:
- Model saving 
- Core metrics: Accuracy, F1, and Log Loss (if possible)
- Classification report + class distribution
- Confusion matrix plot
- ROC and Precision-Recall curves 
- Option to organize results in timestamped subfolders
"""

import os
import joblib
from datetime import datetime
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss


def save_run_outputs(model, X, y_true, y_pred, name, base_output_dir="performance_and_models/knn", use_timestamp_subfolder=True):
    if use_timestamp_subfolder:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, run_id)
    else:
        output_dir = base_output_dir
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    os.makedirs(output_dir, exist_ok=True)

    # Save the trained model
    joblib.dump(model, os.path.join(output_dir, "model.joblib"))

    # Save model parameters if available
    try:
        params = model.get_params()
        with open(os.path.join(output_dir, "model_params.txt"), "w") as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
    except Exception as e:
        print(f"Could not save model parameters: {e}")


    # Computing log loss if probabilities exist and match
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

    # Save classification report
    with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
         f.write(classification_report(y_true, y_pred, digits=4))

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

    # Predict probabilities (if supported)
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)
            # Save raw probabilities
            prob_df = pd.DataFrame(y_proba)
            prob_df.to_csv(os.path.join(output_dir, f"{name}_probabilities.csv"), index=False)

            # ROC Curve (for multiclass with OneVsRest-like interpretation)
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
