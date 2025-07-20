"""
Evaluation wrapper for CNN + MLP models using image and TF-IDF inputs.

- Accepts precomputed or on-the-fly predictions.
- Computes accuracy and macro-F1 score.
- Saves predictions, metrics, and confusion matrix using `save_run_outputs`.
"""

from models.ID_with_TD.CNN_MLP.result_saver import save_run_outputs
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, TensorDataset


def evaluate_model(model, X_img, X_text, y_true, y_pred, name, base_output_dir, batch_size=32):
    """
    Evaluate a CNN + MLP model on image and text inputs.

    Args:
        - model (nn.Module): Trained PyTorch model.
        - X_img (array or tensor): Image inputs of shape [N, 3, 128, 128].
        - X_text (array or tensor): TF-IDF text inputs of shape [N, 3000].
        - y_true (array or tensor): Ground truth class labels.
        - y_pred (array or None): Optional. If None, predictions will be computed.
        - name (str): Identifier for the dataset split ("val", "test", etc.).
        - base_output_dir (str): Directory to save output results.
        - batch_size (int): Batch size for prediction.

    Returns:
        None. Results are printed and saved to disk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Convert to tensors if necessary
    if not torch.is_tensor(X_img):
        X_img = torch.tensor(X_img, dtype=torch.float32)
    if not torch.is_tensor(X_text):
        X_text = torch.tensor(X_text, dtype=torch.float32)
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true, dtype=torch.long)

    dataset = TensorDataset(X_img, X_text, y_true)
    loader = DataLoader(dataset, batch_size=batch_size)

    # If predictions were not provided, compute them in batches
    if y_pred is None:
        print("Computing predictions in batches...")
        all_preds = []
        with torch.no_grad():
            for X_img_batch, X_text_batch, _ in loader:
                X_img_batch = X_img_batch.to(device)
                X_text_batch = X_text_batch.to(device)
                logits = model(X_img_batch, X_text_batch)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
        y_pred = all_preds

    # Compute evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name.capitalize()} Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    # Save outputs (predictions, metrics, confusion matrix)
    save_run_outputs(
        model=model,
        X=(X_img, X_text),
        y_true=y_true,
        y_pred=y_pred,
        name=name,
        base_output_dir=base_output_dir,
        use_timestamp_subfolder=False
    )
