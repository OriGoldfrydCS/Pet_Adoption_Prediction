"""
Evaluation and saving utilities for CNN + LSTM models (mean images + BERT(CLS/full) input).

Includes:
- Cross-entropy loss computation on custom batches.
- Saving model weights, architecture, predictions, classification report, and metrics.
- Confusion matrix visualization and export.

Used for both validation and test set evaluation.
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


def compute_crossentropy_loss(model, X_img, X_text_seq, y_true, batch_size=32):
    """
    Compute average cross-entropy loss for the given model and dataset.

    Args:
        - model (torch.nn.Module): Trained model to evaluate.
        - X_img (np.ndarray or torch.Tensor): Image input, shape [N, C, H, W].
        - X_text_seq (np.ndarray or torch.Tensor): Text input, shape [N, T, 768] or [N, 768].
        - y_true (np.ndarray or torch.Tensor): True class labels, shape [N].
        - batch_size (int): Batch size for evaluation.

    Returns:
        - float: Average cross-entropy loss across the dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # Convert to tensors if needed
    if not torch.is_tensor(X_img):
        X_img = torch.tensor(X_img, dtype=torch.float32)
    if not torch.is_tensor(X_text_seq):
        X_text_seq = torch.tensor(X_text_seq, dtype=torch.float32)
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true, dtype=torch.long)

    dataset = TensorDataset(X_img, X_text_seq, y_true)
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
    Save evaluation results, model, metrics, and visualizations for a run.

    Args:
        - model (torch.nn.Module): Trained model.
        - X (tuple): Tuple of (X_img, X_text_seq) input data.
        - y_true (np.ndarray): Ground truth labels.
        - y_pred (np.ndarray): Model predictions.
        - name (str): Identifier for the dataset split (e.g., "val", "test").
        - base_output_dir (str): Path to main output folder.
        - use_timestamp_subfolder (bool): Whether to save in a timestamped subfolder.

    Saves:
        - model.pt: Model weights.
        - hyperparameters.txt: Input shapes and model architecture.
        - true.csv and pred.csv: Labels.
        - classification_report.txt: Precision/recall/F1 breakdown.
        - metrics.txt: Accuracy, F1, loss.
        - confusion_matrix.png: Heatmap visualization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_img, X_text_seq = X
    output_dir = os.path.join(
        base_output_dir,
        datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp_subfolder else ""
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights
    try:
        model_cpu = model.cpu()
        torch.save(model_cpu.state_dict(), os.path.join(output_dir, "model.pt"))
        model.to(device)  # restore model to device
    except Exception as e:
        print(f"Failed to save model weights: {e}")

    # Save architecture and input shapes
    try:
        with open(os.path.join(output_dir, "hyperparameters.txt"), "w") as f:
            f.write("Model: CNN + LSTM\n")
            f.write(f"Image input shape: {tuple(X_img.shape[1:])}\n")
            f.write(f"Text input shape: {tuple(X_text_seq.shape[1:])} (BERT sequence)\n")
            f.write("Architecture:\n")
            f.write(str(model))
    except Exception as e:
        print(f"Could not save model architecture: {e}")

    # Save true and predicted labels
    pd.Series(y_true).to_csv(os.path.join(output_dir, f"{name}_true.csv"), index=False)
    pd.Series(y_pred).to_csv(os.path.join(output_dir, f"{name}_pred.csv"), index=False)

    # Save classification report
    try:
        with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
            f.write(classification_report(y_true, y_pred, digits=4))
    except Exception as e:
        print(f"Could not save classification report: {e}")

    # Save metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    crossentropy = compute_crossentropy_loss(model, X_img, X_text_seq, y_true)

    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")
        f.write(f"CrossEntropy Loss: {crossentropy:.4f}\n")

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name.capitalize()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"))
    plt.close()

    print(f"\nResults saved in: {output_dir}")