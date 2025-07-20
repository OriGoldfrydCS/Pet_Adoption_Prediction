import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from models.ID_with_TD.CNN_LSTM.result_saver import save_run_outputs

def evaluate_model(model, X_img, X_text_seq, y_true, y_pred=None, name="val", base_output_dir=".", batch_size=32):
    """
    Evaluates a CNN + LSTM model on image and sequence text inputs and saves the results.

    Args:
        model: trained PyTorch model
        X_img: [N, 3, 128, 128] image tensor or NumPy array
        X_text_seq: [N, T, 768] BERT token embeddings
        y_true: [N,] ground truth labels
        y_pred: [N,] predicted labels (optional; if None, will be computed)
        name: string label for the split (e.g., "val", "test")
        base_output_dir: directory to save outputs
        batch_size: evaluation batch size
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Convert to tensors if not already
    if not torch.is_tensor(X_img):
        X_img = torch.tensor(X_img, dtype=torch.float32)
    if not torch.is_tensor(X_text_seq):
        X_text_seq = torch.tensor(X_text_seq, dtype=torch.float32)
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true, dtype=torch.long)

    # Dataset and loader
    dataset = TensorDataset(X_img, X_text_seq, y_true)
    loader = DataLoader(dataset, batch_size=batch_size)

    # Predict if not provided
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

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name.capitalize()} Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    # Save everything
    save_run_outputs(
        model=model,
        X=(X_img, X_text_seq),
        y_true=y_true,
        y_pred=y_pred,
        name=name,
        base_output_dir=base_output_dir,
        use_timestamp_subfolder=False
    )

    return y_pred  # in case you want to reuse it
