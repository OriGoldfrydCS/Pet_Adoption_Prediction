"""
Evaluation and result saving utilities for MLP model with BERT embeddings.

Includes:
- CrossEntropy loss computation for both single and dual-input models 
- Saving predictions, metrics, and visualizations
- Organized output directory with optional timestamp
- Architecture/hyperparameter export for reproducibility
"""

import os
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)


def compute_crossentropy_loss(model, X, y_true):
    """
    Compute cross-entropy loss for a PyTorch model on given inputs and true labels.
    Supports both single-input and dual-input models (X_tokens, X_struct).

    Args:
        - model: Trained PyTorch model.
        - X: Input data (tensor or tuple of tensors).
        - y_true: Ground truth labels (NumPy array).

    Returns:
        - float: Cross-entropy loss value.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    y_tensor = torch.tensor(y_true, dtype=torch.long).to(device)

    # Case: model expects two inputs (tokens + structured)
    if isinstance(X, tuple):
        X_tokens, X_struct = X
        X_tokens = torch.tensor(X_tokens, dtype=torch.float32).to(device)
        X_struct = torch.tensor(X_struct, dtype=torch.float32).to(device)
        logits = model(X_tokens, X_struct)
    else:
        # Case: single input
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        # Handle CNN-specific input shape
        if hasattr(model, 'is_cnn') and model.is_cnn:
            X_tensor = X_tensor.transpose(1, 2)

        logits = model(X_tensor)

    loss = criterion(logits, y_tensor).item()
    return loss


def save_run_outputs(model, X, y_true, y_pred, name, base_output_dir, use_timestamp_subfolder=True):
    """
    Save evaluation outputs from a PyTorch model:
    - Model weights
    - Metrics (accuracy, F1, loss)
    - Classification report and confusion matrix
    - Predictions and ground truth

    Args:
        - model: Trained PyTorch model.
        - X: Input data used for prediction.
        - y_true: Ground truth labels.
        - y_pred: Predicted labels.
        - use_timestamp_subfolder (bool): Whether to create a timestamped run folder.
    """
    from datetime import datetime

    if use_timestamp_subfolder:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, run_id)
    else:
        output_dir = base_output_dir

    os.makedirs(output_dir, exist_ok=True)

    # === Save model weights ===
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    # === Save architecture overview ===
    try:
        with open(os.path.join(output_dir, "hyperparameters.txt"), "w") as f:
            f.write("Model: PyTorch MLP\n")
            f.write(f"Input shape: {tuple(X.shape[1:]) if not isinstance(X, tuple) else 'Tuple input'}\n")
            f.write("Architecture:\n")
            for i, layer in enumerate(model.children()):
                f.write(f"  {i+1}. {layer.__class__.__name__}: {layer}\n")
    except Exception as e:
        print(f"Warning: Could not save hyperparameters.txt: {e}")

    # === Save true and predicted labels ===
    pd.Series(y_true).to_csv(os.path.join(output_dir, f"{name}_true.csv"), index=False)
    pd.Series(y_pred).to_csv(os.path.join(output_dir, f"{name}_pred.csv"), index=False)

    # === Save classification report ===
    with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    # === Save accuracy, F1, and loss ===
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    crossentropy_loss = compute_crossentropy_loss(model, X, y_true)
    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")
        f.write(f"CrossEntropy Loss: {crossentropy_loss:.4f}\n")

    # === Save confusion matrix ===
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
