"""
Evaluation and Output Utilities for Multimodal Models (CNN + MLP + Structured with TF-IDF with mean of all the images per sample)

Provides utility functions for:
- Computing average cross-entropy loss
- Saving model, predictions, metrics, and confusion matrix to disk
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

def compute_crossentropy_loss(model, X_img, X_text, X_struct, y_true, batch_size=32):
    """
    Computes average cross-entropy loss for a given model and dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # Convert inputs to tensors if needed
    X_img = torch.tensor(X_img, dtype=torch.float32) if not torch.is_tensor(X_img) else X_img
    X_text = torch.tensor(X_text, dtype=torch.float32) if not torch.is_tensor(X_text) else X_text
    X_struct = torch.tensor(X_struct, dtype=torch.float32) if not torch.is_tensor(X_struct) else X_struct
    y_tensor = torch.tensor(y_true, dtype=torch.long)

    dataset = TensorDataset(X_img, X_text, X_struct, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size)

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_img_b, x_text_b, x_struct_b, y_b in loader:
            x_img_b = x_img_b.to(device)
            x_text_b = x_text_b.to(device)
            x_struct_b = x_struct_b.to(device)
            y_b = y_b.to(device)

            logits = model(x_img_b, x_text_b, x_struct_b)
            loss = criterion(logits, y_b)

            total_loss += loss.item() * y_b.size(0)
            total_samples += y_b.size(0)

    return total_loss / total_samples


def save_run_outputs(*, model, X_img, X_text, X_struct, y_true, y_pred, name, base_output_dir, use_timestamp_subfolder=True):
    """
    Saves evaluation results for a trained model:
    - Model weights (.pt)
    - Hyperparameters & input shapes (text)
    - Classification report (text)
    - Metrics (accuracy, F1, cross-entropy)
    - True/predicted labels (.csv)
    - Confusion matrix plot (.png)

    Args:
        - model: Trained PyTorch model.
        - X_img, X_text, X_struct: Input tensors or arrays.
        - y_true: True class labels.
        - y_pred: Predicted class labels.
        - name: Prefix for all output files (e.g. "val").
        - use_timestamp_subfolder: If True, creates a subfolder named by current timestamp.
    """
    from datetime import datetime

    # === Prepare output directory ===
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
            f.write("Model: PyTorch Hybrid (CNN + MLP + Structured)\n")
            try:
                img_shape = tuple(X_img.shape[1:]) if hasattr(X_img, "shape") else "Unavailable"
                text_shape = tuple(X_text.shape[1:]) if hasattr(X_text, "shape") else "Unavailable"
                struct_shape = tuple(X_struct.shape[1:]) if hasattr(X_struct, "shape") else "Unavailable"
                f.write(f"Image Input Shape: {img_shape}\n")
                f.write(f"Text Input Shape: {text_shape}\n")
                f.write(f"Structured Input Shape: {struct_shape}\n")
            except Exception as shape_err:
                f.write(f"Input shape error: {shape_err}\n")

            f.write("Architecture:\n")
            for i, layer in enumerate(model.children()):
                f.write(f"  {i+1}. {layer.__class__.__name__}: {layer}\n")
    except Exception as e:
        print(f"Warning: Could not save hyperparameters.txt: {e}")

    # === Save true/predicted labels ===
    pd.Series(y_true).to_csv(os.path.join(output_dir, f"{name}_true.csv"), index=False)
    pd.Series(y_pred).to_csv(os.path.join(output_dir, f"{name}_pred.csv"), index=False)

    # === Save classification report ===
    with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    # === Save metrics ===
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    crossentropy = compute_crossentropy_loss(model, X_img, X_text, X_struct, y_true)

    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")
        f.write(f"CrossEntropy Loss: {crossentropy:.4f}\n")

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

    print(f"\nResults saved in: {output_dir}")
