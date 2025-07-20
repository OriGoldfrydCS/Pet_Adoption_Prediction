"""
Evaluation script for CNN + MLP multimodal model (images + TF-IDF text + structured data).

This function evaluates a trained model by computing predictions and performance metrics
"""

from models.SD_with_TD_with_ID.CNN_MLP.result_saver import save_run_outputs as base_saver
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, TensorDataset


def evaluate_model(*, model, X_img, X_text, X_struct, y_true, y_pred=None,
                   name, base_output_dir, use_timestamp_subfolder=True, batch_size=32):
    """
    Evaluates a multimodal model on given input tensors.
    If y_pred is not provided, it will compute predictions using the model.
    Saves outputs using a standard result_saver function.

    Parameters:
    - model: PyTorch model (CNN + MLP)
    - X_img, X_text, X_struct: input arrays (images, text features, structured data)
    - y_true: true labels
    - y_pred: optional precomputed predictions
    - name: run name (used for saving files)
    - use_timestamp_subfolder: whether to create a unique folder for this run
    - batch_size: prediction batch size
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # === Convert to tensors if needed ===
    if not torch.is_tensor(X_img):
        X_img = torch.tensor(X_img, dtype=torch.float32)
    if not torch.is_tensor(X_text):
        X_text = torch.tensor(X_text, dtype=torch.float32)
    if not torch.is_tensor(X_struct):
        X_struct = torch.tensor(X_struct, dtype=torch.float32)
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true, dtype=torch.long)

    dataset = TensorDataset(X_img, X_text, X_struct, y_true)
    loader = DataLoader(dataset, batch_size=batch_size)

    # === Compute predictions if needed ===
    if y_pred is None:
        print("Computing predictions in batches...")
        all_preds = []
        with torch.no_grad():
            for X_img_b, X_text_b, X_struct_b, _ in loader:
                X_img_b = X_img_b.to(device)
                X_text_b = X_text_b.to(device)
                X_struct_b = X_struct_b.to(device)

                logits = model(X_img_b, X_text_b, X_struct_b)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
        y_pred = all_preds

    # === Compute metrics ===
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name.capitalize()} Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    # === Save results ===
    base_saver(
        model=model,
        X_img=X_img,
        X_text=X_text,
        X_struct=X_struct,
        y_true=y_true,
        y_pred=y_pred,
        name=name,
        base_output_dir=base_output_dir,
        use_timestamp_subfolder=use_timestamp_subfolder
    )
