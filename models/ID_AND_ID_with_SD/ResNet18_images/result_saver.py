"""
This script includes two independent CNN evaluation modules:

1. Resulting a ResNet18 trained on **image data only**
2. Resulting a ResNet18 trained on **image data + structured features**

To run one of them, simply comment out the other.

Each module:
- Receives a trained model and evaluation data
- Computes accuracy, F1 score, and log loss
- Generates a classification report and confusion matrix
- Saves predictions and all evaluation outputs to disk
"""



# === Resulting ResNet18 on image data only ===

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, log_loss
)
from datetime import datetime

def save_run_outputs(model, X, y_true, y_pred, name, base_output_dir, use_timestamp_subfolder=True):
    """
    Save predictions, metrics, and confusion matrix for image-only model.
    """
    if use_timestamp_subfolder:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, run_id)
    else:
        output_dir = base_output_dir
    os.makedirs(output_dir, exist_ok=True)

    # === SKIP MODEL SAVING ===
    print("[INFO] Skipping model saving inside save_run_outputs() to avoid potential CUDA crash.")
    print("[INFO] Save the model manually in training, right after training loop.")

    # === SAVE PREDICTIONS ===
    pd.Series(y_true).to_csv(os.path.join(output_dir, f"{name}_true.csv"), index=False)
    pd.Series(y_pred).to_csv(os.path.join(output_dir, f"{name}_pred.csv"), index=False)

    # === SAVE CLASSIFICATION REPORT ===
    with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    # === SAVE METRICS ===
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")

    # === COMPUTE LOG LOSS (OPTIONAL) ===
    try:
        if model is not None and X is not None:
            model.eval()
            with torch.no_grad():
                device = next(model.parameters()).device

                # Convert to tensor
                if isinstance(X, np.ndarray):
                    X_tensor = torch.tensor(X, dtype=torch.float32)
                elif isinstance(X, torch.Tensor):
                    X_tensor = X.float()
                else:
                    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)

                # Ensure channel format: [B, C, H, W]
                if X_tensor.ndim == 4 and X_tensor.shape[1] != 3:
                    X_tensor = X_tensor.permute(0, 3, 1, 2)

                X_tensor = X_tensor.to(device)

                try:
                    logits = model(X_tensor)
                    y_proba = torch.softmax(logits, dim=1).detach().cpu().numpy()
                except RuntimeError as e:
                    print(f"[FORWARD ERROR] Failed during forward pass: {e}")
                    return

            if y_proba.shape[1] == len(np.unique(y_true)):
                logloss = log_loss(y_true, y_proba)
                with open(os.path.join(output_dir, f"{name}_metrics.txt"), "a") as f:
                    f.write(f"Log Loss: {logloss:.4f}\n")
    except Exception as e:
        print(f"[WARNING] Skipping log loss due to error: {e}")

    # === CONFUSION MATRIX ===
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





# === Resulting ResNet18 with image + structured data ===

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def save_run_outputs(model, X, y_true, y_pred, name, base_output_dir, use_timestamp_subfolder=True, X_struct=None):
    """
    Save predictions, metrics, and confusion matrix for image + structured model.
    """
    # Create output directory
    if use_timestamp_subfolder:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, run_id)
    else:
        output_dir = base_output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights
    if model is not None:
        model_path = os.path.join(output_dir, "model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    # Save predictions and true labels
    pd.Series(y_true).to_csv(os.path.join(output_dir, f"{name}_true.csv"), index=False)
    pd.Series(y_pred).to_csv(os.path.join(output_dir, f"{name}_pred.csv"), index=False)

    # Save classification report
    with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    # Save accuracy and F1 score
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")

    # Try to compute log loss if inputs are valid
    if model is not None and X is not None:
        try:
            model.eval()
            device = next(model.parameters()).device

            # Convert image data to tensor
            X_tensor = torch.tensor(X, dtype=torch.float32) if isinstance(X, np.ndarray) else X.float()
            if len(X_tensor.shape) == 4 and X_tensor.shape[1] != 3:
                X_tensor = X_tensor.permute(0, 3, 1, 2)
            X_tensor = X_tensor.to(device)

            # Convert structured data if available
            if X_struct is not None:
                X_struct_tensor = torch.tensor(X_struct, dtype=torch.float32) if isinstance(X_struct, np.ndarray) else X_struct.float()
                X_struct_tensor = X_struct_tensor.to(device)
                logits = model(X_tensor, X_struct_tensor)
            else:
                logits = model(X_tensor)

            y_proba = torch.softmax(logits, dim=1).cpu().numpy()

            # Compute log loss if shapes match
            if y_proba.shape[0] == len(y_true) and y_proba.shape[1] >= len(np.unique(y_true)):
                logloss = log_loss(y_true, y_proba)
                with open(os.path.join(output_dir, f"{name}_metrics.txt"), "a") as f:
                    f.write(f"Log Loss: {logloss:.4f}\n")
            else:
                print(f"Skipping log loss due to shape mismatch: y_proba={y_proba.shape}, y_true={len(y_true)}")

        except Exception as e:
            print(f"[WARNING] Log loss skipped due to error: {e}")

    # Save confusion matrix as heatmap
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
