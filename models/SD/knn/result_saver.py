"""
Saving module for KNN evaluation results.

This function handles all output-related tasks after evaluating a model, including:
- Saving the trained model
- Writing classification reports and basic metrics (accuracy, F1, log loss)
- Plotting and saving confusion matrix, ROC and PR curves
- Computes training metrics if X_train and y_train are passed

Used after evaluating on either validation or test set.
"""

import os
import joblib
from datetime import datetime
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc, log_loss
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def save_run_outputs(
    model,
    X,
    y_true,
    y_pred,
    name,
    base_output_dir="performance_and_models/knn",
    use_timestamp_subfolder=True,
    X_train=None,
    y_train=None
):
    """
    Saves model outputs, evaluation metrics, and plots.

    Parameters:
    - model: Trained scikit-learn model 
    - X: Feature matrix used for prediction
    - y_true: True labels
    - y_pred: Predicted labels
    - name (str): Identifier for the current run ('val' or 'test')
    - base_output_dir (str): Directory where outputs will be saved
    - use_timestamp_subfolder (bool): Whether to add a timestamped folder inside base_output_dir
    - X_train, y_train: Optional training data to compute training accuracy/log loss

    Returns:
    - None (saves results)
    """

    # Create output directory
    if use_timestamp_subfolder:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, run_id)
    else:
        output_dir = base_output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Save model
    joblib.dump(model, os.path.join(output_dir, "model.joblib"))

    # Save model hyperparameters (if available)
    try:
        params = model.get_params()
        with open(os.path.join(output_dir, "model_params.txt"), "w") as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
    except Exception as e:
        print(f"Could not save model parameters: {e}")

    # Save classification report
    with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    # Compute test metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    # Try computing log loss
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

    # Also compute train accuracy/log loss if data provided
    train_acc = None
    train_logloss = None
    if X_train is not None and y_train is not None:
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        if hasattr(model, "predict_proba"):
            try:
                y_train_proba = model.predict_proba(X_train)
                if y_train_proba.shape[1] == len(np.unique(y_train)):
                    train_logloss = log_loss(y_train, y_train_proba)
            except Exception as e:
                print(f"Error computing train log loss: {e}")

    # Save metrics
    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Test F1 (macro): {f1:.4f}\n")
        if logloss is not None:
            f.write(f"Test Log Loss: {logloss:.4f}\n")
        if train_acc is not None:
            f.write(f"Train Accuracy: {train_acc:.4f}\n")
        if train_logloss is not None:
            f.write(f"Train Log Loss: {train_logloss:.4f}\n")

    # Save class distributions
    pd.Series(y_true).value_counts().to_csv(os.path.join(output_dir, f"{name}_true_distribution.txt"))
    pd.Series(y_pred).value_counts().to_csv(os.path.join(output_dir, f"{name}_pred_distribution.txt"))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name.capitalize()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"))
    plt.close()

    # Probability-based plots and files
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)

            # Save raw probabilities
            prob_df = pd.DataFrame(y_proba)
            prob_df.to_csv(os.path.join(output_dir, f"{name}_probabilities.csv"), index=False)

            # ROC Curve per class
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

            # PR Curve per class
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
