"""
This script evaluates a trained multimodal model (CLIP + text + structured).
It handles:
- Batch-wise inference on GPU/CPU
- Accuracy and macro F1 score computation
- Saving all results using `save_run_outputs`

Used for both validation and test evaluation after training.
"""

from models.SD_with_TD_with_ID.CLIP_MLP.result_saver import save_run_outputs as base_saver
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def evaluate_model(*, model, X_img, X_text, X_struct, y_true, y_pred=None,
                   name, base_output_dir, use_timestamp_subfolder=True, batch_size=32):
    """
    Evaluates a trained model on given data.
    
    Parameters:
        - model: Trained PyTorch model
        - X_img, X_text, X_struct: Input features
        - y_true: True labels
        - y_pred: Optional predictions (if already computed)
        - name: Identifier for saving files ("val" or "test")
        - base_output_dir: Directory to store evaluation results
        - use_timestamp_subfolder: Whether to save outputs in timestamped subdirectory
        - batch_size: Batch size for inference

    Outputs:
        Prints accuracy and F1 score, saves results via `result_saver`
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # === Convert inputs to torch tensors if needed ===
    X_img = torch.tensor(X_img, dtype=torch.float32) if not torch.is_tensor(X_img) else X_img
    X_text = torch.tensor(X_text, dtype=torch.float32) if not torch.is_tensor(X_text) else X_text
    X_struct = torch.tensor(X_struct, dtype=torch.float32) if not torch.is_tensor(X_struct) else X_struct
    y_tensor = torch.tensor(y_true, dtype=torch.long) if not torch.is_tensor(y_true) else y_true

    dataset = TensorDataset(X_img, X_text, X_struct, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size)

    # === Generate predictions if not provided ===
    if y_pred is None:
        print("Computing predictions in batches...")
        y_pred = []
        with torch.no_grad():
            for x_img_b, x_text_b, x_struct_b, _ in loader:
                x_img_b = x_img_b.to(device)
                x_text_b = x_text_b.to(device)
                x_struct_b = x_struct_b.to(device)

                logits = model(x_img_b, x_text_b, x_struct_b)
                preds = torch.argmax(logits, dim=1)
                y_pred.extend(preds.cpu().numpy())

    # === Convert labels to numpy ===
    y_true_np = y_tensor.cpu().numpy() if torch.is_tensor(y_tensor) else y_true

    # === Print evaluation metrics ===
    acc = accuracy_score(y_true_np, y_pred)
    f1 = f1_score(y_true_np, y_pred, average="macro")
    print(f"{name.capitalize()} Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    # === Save results using the central saver ===
    base_saver(
        model=model,
        X_img=X_img,
        X_text=X_text,
        X_struct=X_struct,
        y_true=y_true_np,
        y_pred=y_pred,
        name=name,
        base_output_dir=base_output_dir,
        use_timestamp_subfolder=use_timestamp_subfolder
    )
