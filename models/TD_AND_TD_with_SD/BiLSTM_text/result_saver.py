"""
Evaluation utilities for BiLSTM model reporting and saving results.
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
    confusion_matrix)

def compute_crossentropy_loss(model, X, y_true):
    """
    Computes the cross-entropy loss for a model given input X and true labels.

    Args:
        model: Trained PyTorch model
        X: Input data (tensor or tuple of tensors)
        y_true: True labels (array-like)

    Returns:
        Cross-entropy loss (float)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    y_tensor = torch.tensor(y_true, dtype=torch.long).to(device)

    if isinstance(X, tuple):
        X_seq, X_struct = X
        X_seq = torch.tensor(X_seq, dtype=torch.float32).to(device)
        X_struct = torch.tensor(X_struct, dtype=torch.float32).to(device)
        logits = model(X_seq, X_struct)
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(X_tensor)

    loss = criterion(logits, y_tensor).item()
    return loss


def summarize_model_layers(model, f, indent=0):
    """
    Recursively writes model architecture to a file-like object.

    Args:
        - model: PyTorch model
        - f: File object
        - indent: Current indentation level
    """
    for name, module in model.named_children():
        f.write("  " * indent + f"{name}: {module.__class__.__name__}({module})\n")
        summarize_model_layers(module, f, indent + 1)


def save_run_outputs(model, X, y_true, y_pred, name, base_output_dir, use_timestamp_subfolder=True):
    """
    Saves all outputs from a model evaluation run:
    - Weights
    - Hyperparameters
    - Predictions
    - Classification report
    - Metrics (Accuracy, F1, CrossEntropy)
    - Confusion matrix plot
    - Softmax probabilities (optional)

    Args:
        - model: Trained model
        - X: Input data (tensor or tuple of tensors)
        - y_true: Ground-truth labels
        - y_pred: Predicted labels
        - use_timestamp_subfolder: Whether to save in a unique timestamped folder
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

    # Save hyperparameters and architecture
    try:
        with open(os.path.join(output_dir, "hyperparameters.txt"), "w") as f:
            f.write(f"Model: {model.__class__.__name__}\n")
            f.write(f"Input shape: {tuple(X.shape) if not isinstance(X, tuple) else 'multiple'}\n")
            f.write("Architecture:\n")
            summarize_model_layers(model, f)
    except Exception as e:
        print(f"Warning: Could not save hyperparameters.txt: {e}")

    # Save predictions and labels
    pd.Series(y_true).to_csv(os.path.join(output_dir, f"{name}_true.csv"), index=False)
    pd.Series(y_pred).to_csv(os.path.join(output_dir, f"{name}_pred.csv"), index=False)

    # Save classification report
    with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    # Save metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    crossentropy = compute_crossentropy_loss(model, X, y_true)
    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")
        f.write(f"CrossEntropy Loss: {crossentropy:.4f}\n")

    # Save confusion matrix as image
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name.capitalize()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"))
    plt.close()

    # Optional: Save softmax probabilities
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        if isinstance(X, tuple):
            X_seq, X_struct = X
            X_seq = torch.tensor(X_seq, dtype=torch.float32).to(device)
            X_struct = torch.tensor(X_struct, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = model(X_seq, X_struct)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            if hasattr(model, 'is_cnn') and model.is_cnn:
                X_tensor = X_tensor.transpose(1, 2)
            with torch.no_grad():
                logits = model(X_tensor)

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        np.savetxt(os.path.join(output_dir, f"{name}_probs.csv"), probs, delimiter=",", fmt="%.5f")
    except Exception as e:
        print(f"Warning: Could not save probabilities: {e}")

    print(f"Results saved in: {output_dir}")
