"""
Evaluation utilities for CNN + MLP models (image + TF-IDF inputs).

Includes:
- Batch-wise cross-entropy loss calculation.
- Saving model weights, input shapes, predictions, metrics, and plots.
- Outputs are saved to a timestamped directory.
"""

import os
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from sklearn.metrics import (
    log_loss, classification_report, accuracy_score, f1_score,
    confusion_matrix
)


def compute_crossentropy_loss(model, X_img, X_text, y_true, batch_size=32):
    """
    Compute average cross-entropy loss over dataset in batches.

    Args:
        - model (nn.Module): Trained PyTorch model.
        - X_img (np.ndarray or torch.Tensor): Image input, shape [N, 3, 128, 128].
        - X_text (np.ndarray or torch.Tensor): Text input, shape [N, 3000].
        - y_true (np.ndarray): True labels.
        - batch_size (int): Batch size for evaluation.

    Returns:
        float: Average cross-entropy loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # Convert to tensors if needed
    X_img = torch.tensor(X_img, dtype=torch.float32) if not torch.is_tensor(X_img) else X_img
    X_text = torch.tensor(X_text, dtype=torch.float32) if not torch.is_tensor(X_text) else X_text
    y_tensor = torch.tensor(y_true, dtype=torch.long)

    dataset = TensorDataset(X_img, X_text, y_tensor)
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


def save_run_outputs(model, X_img, X_text, y_true, y_pred, name, base_output_dir, use_timestamp_subfolder=True):
    """
    Save model, predictions, evaluation metrics, and confusion matrix to disk.

    Args:
        - model (nn.Module): Trained model to be saved.
        - X_img (np.ndarray or torch.Tensor): Image inputs.
        - X_text (np.ndarray or torch.Tensor): Text inputs (TF-IDF).
        - y_true (np.ndarray): Ground truth labels.
        - y_pred (np.ndarray): Predicted labels.
        - name (str): Identifier for output (e.g., "val", "test").
        - base_output_dir (str): Path to base output directory.
        - use_timestamp_subfolder (bool): Whether to create a timestamped folder inside the base directory.
    """
    from datetime import datetime

    # === Create output directory ===
    if use_timestamp_subfolder:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, run_id)
    else:
        output_dir = base_output_dir
    os.makedirs(output_dir, exist_ok=True)

    # === Save model weights ===
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    # === Save architecture and input shapes ===
    try:
        with open(os.path.join(output_dir, "hyperparameters.txt"), "w") as f:
            f.write("Model: PyTorch Hybrid (CNN + MLP)\n")
            try:
                img_shape = tuple(X_img.shape[1:]) if hasattr(X_img, "shape") else "Unavailable"
                text_shape = tuple(X_text.shape[1:]) if hasattr(X_text, "shape") else "Unavailable"
                f.write(f"Image Input Shape: {img_shape}\n")
                f.write(f"Text Input Shape: {text_shape}\n")
            except Exception as shape_err:
                f.write(f"Input shape error: {shape_err}\n")

            f.write("Architecture:\n")
            for i, layer in enumerate(model.children()):
                f.write(f"  {i+1}. {layer.__class__.__name__}: {layer}\n")
    except Exception as e:
        print(f"Warning: Could not save hyperparameters.txt: {e}")

    # === Save predictions and labels ===
    pd.Series(y_true).to_csv(os.path.join(output_dir, f"{name}_true.csv"), index=False)
    pd.Series(y_pred).to_csv(os.path.join(output_dir, f"{name}_pred.csv"), index=False)

    # === Save classification report ===
    with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    # === Save metrics ===
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    crossentropy = compute_crossentropy_loss(model, X_img, X_text, y_true)

    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")
        f.write(f"CrossEntropy Loss: {crossentropy:.4f}\n")

    # === Save confusion matrix plot ===
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
