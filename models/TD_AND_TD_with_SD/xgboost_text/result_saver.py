"""
Utility for saving evaluation results of XgBoost models (with or without structured data).

Includes:
- Saving model weights 
- Storing predictions, metrics, and plots
- Computes accuracy, F1, and log loss 
- Generates confusion matrix visualization
- Works with both torch and numpy input
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, log_loss
)
from datetime import datetime

def save_run_outputs(model, X, y_true, y_pred, name, base_output_dir, use_timestamp_subfolder=True):
    """
    Save model, predictions, metrics, and plots to a specified directory

    Args:
        - model: Trained model (PyTorch or sklearn-style with `predict_proba`)
        - X: Input features (numpy array or torch tensor)
        - y_true: True labels
        - y_pred: Predicted labels
        - name: Identifier for this evaluation run ('val', 'test')
    """
    if use_timestamp_subfolder:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, run_id)
    else:
        output_dir = base_output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Save PyTorch model if provided
    if model is not None:
        model_path = os.path.join(output_dir, "model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    # Save predictions and labels
    pd.Series(y_true).to_csv(os.path.join(output_dir, f"{name}_true.csv"), index=False)
    pd.Series(y_pred).to_csv(os.path.join(output_dir, f"{name}_pred.csv"), index=False)

    # Classification report
    with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    # Accuracy & F1
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")

    # Compute log loss if possible
    try:
        y_proba = None

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
        elif model is not None:
            with torch.no_grad():
                if isinstance(X, np.ndarray):
                    X_tensor = torch.tensor(X, dtype=torch.float32)
                elif isinstance(X, torch.Tensor):
                    X_tensor = X.float()
                else:
                    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)

                # Permute if needed (e.g., for images)
                if len(X_tensor.shape) == 4 and X_tensor.shape[1] != 3:
                    X_tensor = X_tensor.permute(0, 3, 1, 2)

                device = next(model.parameters()).device
                X_tensor = X_tensor.to(device)

                model.eval()
                logits = model(X_tensor)
                y_proba = torch.softmax(logits, dim=1).cpu().numpy()

        y_true_cpu = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else np.array(y_true)

        if y_proba is not None and y_proba.shape[0] == len(y_true_cpu):
            try:
                logloss = log_loss(y_true_cpu, y_proba)
                with open(os.path.join(output_dir, f"{name}_metrics.txt"), "a") as f:
                    f.write(f"Log Loss: {logloss:.4f}\n")
            except ValueError as e:
                print(f"Skipping log loss due to mismatch: {e}")
        else:
            print("Log loss skipped: shape mismatch or y_proba is None.")

    except Exception as e:
        print(f"Skipping log loss due to error: {e}")

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name.capitalize()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"))
    plt.close()

    print(f"Results saved in: {output_dir}")
