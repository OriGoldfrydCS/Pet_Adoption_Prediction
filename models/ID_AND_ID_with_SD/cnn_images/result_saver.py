"""
This script includes two independent CNN evaluation modules:

1. Resulting a CNN trained on **image data only**
2. Resulting a CNN trained on **image data + structured features**

To run one of them, simply comment out the other.

Each module:
- Receives a trained model and evaluation data
- Computes accuracy, F1 score, and log loss
- Generates a classification report and confusion matrix
- Saves predictions and all evaluation outputs to disk
"""


# === Resulting CNN on image data only ===

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, log_loss
)
from datetime import datetime

def save_run_outputs(model, X, y_true, y_pred, name, base_output_dir, use_timestamp_subfolder=True):
    if use_timestamp_subfolder:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, run_id)
    else:
        output_dir = base_output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Save PyTorch model
    if model is not None:
        model_path = os.path.join(output_dir, "model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    # Save predictions and true values
    pd.Series(y_true).to_csv(os.path.join(output_dir, f"{name}_true.csv"), index=False)
    pd.Series(y_pred).to_csv(os.path.join(output_dir, f"{name}_pred.csv"), index=False)

    # Save classification report
    with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    # Save accuracy & F1
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")

    # Compute log loss
    y_proba = None
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)

        elif model is not None:
            with torch.no_grad():
                if isinstance(X, np.ndarray):
                    X_tensor = torch.tensor(X, dtype=torch.float32)
                elif isinstance(X, torch.Tensor):
                    X_tensor = X.float()
                else:
                    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)

                if len(X_tensor.shape) == 4 and X_tensor.shape[1] != 3:
                    X_tensor = X_tensor.permute(0, 3, 1, 2)

                device = next(model.parameters()).device
                X_tensor = X_tensor.to(device)

                model.eval()
                logits = model(X_tensor)
                y_proba = torch.softmax(logits, dim=1).cpu().numpy()

        y_true_cpu = (
            y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else np.array(y_true)
        )

        print("Log loss debug:")
        print("y_proba shape:", y_proba.shape if y_proba is not None else "None")
        print("y_true shape:", y_true_cpu.shape)
        print("y_proba sample:", y_proba[:2] if y_proba is not None else "None")
        print("y_true unique:", np.unique(y_true_cpu))

        if y_proba is not None and y_proba.shape[0] == len(y_true_cpu):
            logloss = log_loss(y_true_cpu, y_proba)
            with open(os.path.join(output_dir, f"{name}_metrics.txt"), "a") as f:
                f.write(f"Log Loss: {logloss:.4f}\n")
        else:
            print("Log loss skipped: shape mismatch or y_proba missing.")

    except Exception as e:
        print(f"Skipping log loss due to error: {e}")

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

    print(f"Results saved in: {output_dir}")




# === Resulting CNN with image + structured data ===

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, log_loss
)
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_run_outputs(model, X, y_true, y_pred, name, base_output_dir, use_timestamp_subfolder=True):
    if use_timestamp_subfolder:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, run_id)
    else:
        output_dir = base_output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Save model
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

    # Accuracy and F1
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")

    # === Compute log loss (for dual-input models) ===
    try:
        y_true_cpu = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else np.array(y_true)

        if isinstance(X, tuple) and len(X) == 2:
            x_img, x_struct = X

            x_img = torch.tensor(x_img, dtype=torch.float32) if not torch.is_tensor(x_img) else x_img
            x_struct = torch.tensor(x_struct, dtype=torch.float32) if not torch.is_tensor(x_struct) else x_struct

            if x_img.ndim == 4 and x_img.shape[1] != 3:
                x_img = x_img.permute(0, 3, 1, 2).contiguous()  # 💡 FIXED: ensure safe memory layout

            x_img = x_img.to(device)
            x_struct = x_struct.to(device)

            model.eval()
            with torch.no_grad():
                logits = model(x_img, x_struct)
                y_proba = torch.softmax(logits, dim=1).cpu().numpy()

            if y_proba.shape[0] == len(y_true_cpu):
                logloss = log_loss(y_true_cpu, y_proba)
                with open(os.path.join(output_dir, f"{name}_metrics.txt"), "a") as f:
                    f.write(f"Log Loss: {logloss:.4f}\n")
            else:
                print("Log loss skipped: shape mismatch.")
        else:
            print("Log loss skipped: X is not a dual input (tuple).")

    except Exception as e:
        print(f"Skipping log loss due to error: {e}")

    # Confusion matrix
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
