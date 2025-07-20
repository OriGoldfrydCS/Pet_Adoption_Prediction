"""
Multimodal classification model: CNN + Attention + MLP

This script handles evaluation and result logging for model trained on:
- Image sequences 
- TF-IDF text vectors
- Structured data

Key functionalities:
- Cross-entropy loss computation with image padding and attention mask
- Metrics: accuracy, F1, and classification report
- Saving model weights, predictions, metrics, and confusion matrix
- Logging architecture and input shapes to a timestamped directory
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
    Computes the average cross-entropy loss over a dataset with multiple images per sample.
    Handles padding and masking for attention-based models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # Convert to tensors if needed
    if not torch.is_tensor(X_text):
        X_text = torch.tensor(X_text, dtype=torch.float32)
    if not torch.is_tensor(X_struct):
        X_struct = torch.tensor(X_struct, dtype=torch.float32)
    y_tensor = torch.tensor(y_true, dtype=torch.long)

    def pad_image_sequences(image_lists, max_len=5):
        """
        Pads image sequences to a fixed length and returns a boolean mask.
        """
        batch_size = len(image_lists)
        C, H, W = image_lists[0][0].shape

        padded = torch.zeros((batch_size, max_len, C, H, W), dtype=torch.float32)
        mask = torch.ones((batch_size, max_len), dtype=torch.bool)

        for i, seq in enumerate(image_lists):
            truncated = seq[:max_len]
            truncated_tensors = [
                torch.from_numpy(img).float() if isinstance(img, np.ndarray) else img.float()
                for img in truncated
            ]
            padded[i, :len(truncated_tensors)] = torch.stack(truncated_tensors)
            mask[i, :len(truncated_tensors)] = False

        return padded, mask

    # Create batches manually for image padding
    indices = list(range(len(X_text)))
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in batches:
            batch_imgs = [X_img[i] for i in batch]
            padded_imgs, mask = pad_image_sequences(batch_imgs)
            x_text_b = X_text[batch].to(device)
            x_struct_b = X_struct[batch].to(device)
            y_b = y_tensor[batch].to(device)
            padded_imgs = padded_imgs.to(device)
            mask = mask.to(device)

            logits = model(padded_imgs, x_text_b, x_struct_b, image_mask=mask)
            loss = criterion(logits, y_b)

            total_loss += loss.item() * y_b.size(0)
            total_samples += y_b.size(0)

    return total_loss / total_samples


def save_run_outputs(*, model, X_img, X_text, X_struct, y_true, y_pred, name, base_output_dir, use_timestamp_subfolder=True):
    """
    Saves model weights, predictions, metrics, and confusion matrix

    Parameters:
        - model: trained PyTorch model
        - X_img, X_text, X_struct: input data
        - y_true, y_pred: true and predicted labels
        - use_timestamp_subfolder: whether to create a new subfolder per run
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
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    # === Save input shape info & model layers ===
    try:
        with open(os.path.join(output_dir, "hyperparameters.txt"), "w") as f:
            f.write("Model: PyTorch Hybrid (CNN + MLP + Structured)\n")
            try:
                img_info = f"{len(X_img)} samples, {len(X_img[0])} images per sample" if isinstance(X_img, list) else "Unavailable"
                text_shape = tuple(X_text.shape[1:]) if hasattr(X_text, "shape") else "Unavailable"
                struct_shape = tuple(X_struct.shape[1:]) if hasattr(X_struct, "shape") else "Unavailable"
                f.write(f"Image Input Info: {img_info}\n")
                f.write(f"Text Input Shape: {text_shape}\n")
                f.write(f"Structured Input Shape: {struct_shape}\n")
            except Exception as shape_err:
                f.write(f"Input shape error: {shape_err}\n")

            f.write("Architecture:\n")
            for i, layer in enumerate(model.children()):
                f.write(f"  {i + 1}. {layer.__class__.__name__}: {layer}\n")
    except Exception as e:
        print(f"Warning: Could not save hyperparameters.txt: {e}")

    # === Save predictions and labels ===
    pd.Series(y_true).to_csv(os.path.join(output_dir, f"{name}_true.csv"), index=False)
    pd.Series(y_pred).to_csv(os.path.join(output_dir, f"{name}_pred.csv"), index=False)

    # === Classification report ===
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

    # === Confusion matrix plot ===
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
