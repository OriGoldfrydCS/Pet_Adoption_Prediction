"""
This script handles evaluation and output saving for a multimodal model that uses:
- CLIP image embeddings
- CLIP text embeddings
- Structured numeric features

It provides:
- Cross-entropy loss computation
- Accuracy, F1, and classification report
- Saving model weights, predictions, metrics, and confusion matrix plots
- Structured logging in timestamped directories
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
    Computes average cross-entropy loss over the given dataset using the trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # Convert to tensors if needed
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
    Saves all evaluation outputs for a trained model, including:
    - Model weights
    - Input shapes and architecture
    - Predictions and ground truth
    - Classification report and metrics
    - Confusion matrix heatmap

    Parameters:
        - model: Trained PyTorch model
        - X_img, X_text, X_struct: Input arrays/tensors
        - y_true, y_pred: True and predicted labels
        - name: 'val' or 'test' (used in filenames)
        - base_output_dir: Root directory to save results
        - use_timestamp_subfolder: Whether to create a timestamped subfolder
    """
    from datetime import datetime

    # === Output directory ===
    if use_timestamp_subfolder:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, run_id)
    else:
        output_dir = base_output_dir
    os.makedirs(output_dir, exist_ok=True)

    # === Save model weights ===
    try:
        model_cpu = model.to('cpu')
        torch.save(model_cpu.state_dict(), os.path.join(output_dir, "model.pt"))
    except Exception as e:
        print(f"Warning: Failed to save model weights: {e}")

    # === Save input shapes and model architecture ===
    try:
        with open(os.path.join(output_dir, "hyperparameters.txt"), "w") as f:
            f.write("Model: CLIP + MLP + Structured Input\n")

            img_shape = tuple(X_img.shape[1:]) if hasattr(X_img, "shape") else "Unavailable"
            text_shape = tuple(X_text.shape[1:]) if hasattr(X_text, "shape") else "Unavailable"
            struct_shape = tuple(X_struct.shape[1:]) if hasattr(X_struct, "shape") else "Unavailable"
            f.write(f"CLIP Image Embedding Shape:  {img_shape}\n")
            f.write(f"CLIP Text Embedding Shape:   {text_shape}\n")
            f.write(f"Structured Input Shape:      {struct_shape}\n\n")

            f.write("Model Architecture:\n")
            f.write(str(model))
    except Exception as e:
        print(f"Warning: Could not save model architecture: {e}")

    # === Save predictions ===
    pd.Series(y_true).to_csv(os.path.join(output_dir, f"{name}_true.csv"), index=False)
    pd.Series(y_pred).to_csv(os.path.join(output_dir, f"{name}_pred.csv"), index=False)

    # === Save classification report ===
    try:
        with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
            f.write(classification_report(y_true, y_pred, digits=4))
    except Exception as e:
        print(f"Warning: Could not save classification report: {e}")

    # === Save metrics ===
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    crossentropy = compute_crossentropy_loss(model, X_img, X_text, X_struct, y_true)

    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")
        f.write(f"CrossEntropy Loss: {crossentropy:.4f}\n")

    # === Save confusion matrix plot ===
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name.capitalize()} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save confusion matrix plot: {e}")

    print(f"\nResults saved in: {output_dir}")
