"""
Evaluation and output saving utilities for CNN model on BERT token embeddings + structured data.
Includes:
- Cross-entropy loss computation
- Saving model predictions and performance metrics
- Confusion matrix plotting and directory management
"""

import os
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc, log_loss
)


def compute_crossentropy_loss(model, X, y_true):
    """
    Computes CrossEntropy loss for a trained model on given input and true labels.

    Args:
        - model (nn.Module): Trained PyTorch model
        - X (Tensor or tuple): Input features (single tensor or (tokens, structured))
        - y_true (array-like): Ground truth labels

    Returns:
        - float: CrossEntropy loss
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    y_tensor = torch.tensor(y_true, dtype=torch.long).to(device)

    if isinstance(X, tuple):
        # For models with two inputs
        X_tokens, X_struct = X
        X_tokens = torch.tensor(X_tokens, dtype=torch.float32).to(device)
        X_struct = torch.tensor(X_struct, dtype=torch.float32).to(device)
        logits = model(X_tokens, X_struct)
    else:
        # For models with single input
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        if hasattr(model, 'is_cnn') and model.is_cnn:
            X_tensor = X_tensor.transpose(1, 2)  # Handle CNN expected shape

        logits = model(X_tensor)

    loss = criterion(logits, y_tensor).item()
    return loss


def save_run_outputs(model, X, y_true, y_pred, name, base_output_dir, use_timestamp_subfolder=True):
    """
    Saves model, predictions, metrics, and confusion matrix to output directory.

    Args:
        - model (nn.Module): Trained PyTorch model
        - X (Tensor or tuple): Input data used for prediction
        - y_true (array-like): Ground truth labels
        - y_pred (array-like): Model predictions
        - name (str): Prefix name for saved files
        - base_output_dir (str): Base directory for saving outputs
        - use_timestamp_subfolder (bool): Whether to create a timestamped subfolder
    """
    from datetime import datetime

    # Create output directory
    if use_timestamp_subfolder:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, run_id)
    else:
        output_dir = base_output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    # Save architecture summary
    try:
        with open(os.path.join(output_dir, "hyperparameters.txt"), "w") as f:
            f.write("Model: PyTorch\n")
            if not isinstance(X, tuple):
                f.write(f"Input shape: {tuple(X.shape[1:])}\n")
            f.write("Architecture:\n")
            for i, layer in enumerate(model.children()):
                f.write(f"  {i+1}. {layer.__class__.__name__}: {layer}\n")
    except Exception as e:
        print(f"Warning: Could not save hyperparameters.txt: {e}")

    # Save predictions and labels
    pd.Series(y_true).to_csv(os.path.join(output_dir, f"{name}_true.csv"), index=False)
    pd.Series(y_pred).to_csv(os.path.join(output_dir, f"{name}_pred.csv"), index=False)

    # Save classification report
    with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    # Save metrics summary
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    crossentropy_loss = compute_crossentropy_loss(model, X, y_true)
    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")
        f.write(f"CrossEntropy Loss: {crossentropy_loss:.4f}\n")

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

    print(f"Results saved in: {output_dir}")
