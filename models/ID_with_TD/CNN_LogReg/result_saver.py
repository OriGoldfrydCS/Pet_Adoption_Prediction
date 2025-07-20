"""
Evaluation utilities for image + text model -> CNN + LogReg.

Includes:
- Cross-entropy loss computation on datasets.
- Saving predictions, metrics, model weights, and plots.
- Output is saved to a timestamped directory under a given base path.
"""

import os
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, confusion_matrix
)
from datetime import datetime


def compute_crossentropy_loss(model, X_img, X_text, y_true, batch_size=32):
    """
    Compute average cross-entropy loss for a given model and dataset.

    Args:
        - model (torch.nn.Module): Trained model.
        - X_img (np.ndarray or tensor): Image inputs.
        - X_text (np.ndarray or tensor): Text inputs (TF-IDF).
        - y_true (np.ndarray or tensor): Ground truth labels.
        - batch_size (int): Batch size for evaluation.

    Returns:
        - float: Average cross-entropy loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # Ensure all inputs are tensors
    if not torch.is_tensor(X_img):
        X_img = torch.tensor(X_img, dtype=torch.float32)
    if not torch.is_tensor(X_text):
        X_text = torch.tensor(X_text, dtype=torch.float32)
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true, dtype=torch.long)

    dataset = TensorDataset(X_img, X_text, y_true)
    loader = DataLoader(dataset, batch_size=batch_size)

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_img_b, x_text_b, y_b in loader:
            x_img_b = x_img_b.to(device)
            x_text_b = x_text_b.to(device)
            y_b = y_b.to(device)

            logits = model(x_img_b, x_text_b)
            loss = criterion(logits, y_b)

            total_loss += loss.item() * y_b.size(0)
            total_samples += y_b.size(0)

    return total_loss / total_samples


def save_run_outputs(model, X, y_true, y_pred, name, base_output_dir, use_timestamp_subfolder=True):
    """
    Save model, predictions, evaluation metrics, and confusion matrix.

    Args:
        - model (torch.nn.Module): Trained model.
        - X (tuple): Tuple (X_img, X_text).
        - y_true (np.ndarray): Ground truth labels.
        - y_pred (np.ndarray): Predicted labels.
        - name (str): Name prefix for files.
        - base_output_dir (str): Base directory to save outputs.
        - use_timestamp_subfolder (bool): Whether to create timestamped subfolder.
    """
    X_img, X_text = X

    # Create output directory (with timestamp if needed)
    output_dir = os.path.join(
        base_output_dir,
        datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp_subfolder else ""
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    # Save architecture & input shape info
    try:
        with open(os.path.join(output_dir, "hyperparameters.txt"), "w") as f:
            f.write("Model: CNN + LogReg/BiLSTM\n")
            f.write(f"Image input shape: {tuple(X_img.shape[1:])}\n")
            f.write(f"Text input shape: {tuple(X_text.shape[1:])}\n")
            f.write("Architecture:\n")
            f.write(str(model))
    except Exception as e:
        print(f"Could not save hyperparameters.txt: {e}")

    # Save predictions and ground truth labels
    pd.Series(y_true).to_csv(os.path.join(output_dir, f"{name}_true.csv"), index=False)
    pd.Series(y_pred).to_csv(os.path.join(output_dir, f"{name}_pred.csv"), index=False)

    # Save classification report as text
    with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    # Save basic metrics (acc, f1, loss)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    crossentropy = compute_crossentropy_loss(model, X_img, X_text, y_true)

    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")
        f.write(f"CrossEntropy Loss: {crossentropy:.4f}\n")

    # Save confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name.capitalize()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"))
    plt.close()

    print(f"\nResults saved in: {output_dir}")
