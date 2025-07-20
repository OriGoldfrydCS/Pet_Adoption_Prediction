"""
Multimodal classification model: CNN + Attention + MLP

Evaluation script for multimodal model with image sequences, TF-IDF text, and structured data.

Key responsibilities:
- Pads image sequences and creates attention masks
- Handles batch-wise inference with optional precomputed predictions
- Computes accuracy and macro F1 score
- Delegates saving results (metrics, confusion matrix, model weights) to `result_saver`
"""

from models.SD_with_TD_with_ID.CNN_att_MLP.result_saver import save_run_outputs as base_saver
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader
import numpy as np


def pad_image_sequences(image_lists, max_len=5):
    """
    Pads image sequences to a fixed length and generates a mask.
    Args:
        - image_lists: list of lists of image tensors
        - max_len: maximum number of images per sample
    Returns:
        - padded: tensor of shape [B, max_len, C, H, W]
        - mask: tensor of shape [B, max_len], True = padded
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
        mask[i, :len(truncated_tensors)] = False  # valid images marked as False

    return padded, mask


def evaluate_model(*, model, X_img, X_text, X_struct, y_true, y_pred=None,
                   name, base_output_dir, use_timestamp_subfolder=True, batch_size=32):
    """
    Evaluates a trained model on validation/test set and saves all relevant outputs.

    Args:
        - model: trained PyTorch model
        - X_img: list of lists of image tensors or np arrays
        - X_text: TF-IDF vectors
        - X_struct: structured features
        - y_true: true labels
        - y_pred: if provided, skips prediction and only evaluates
        - use_timestamp_subfolder: whether to create a new subfolder
        - batch_size: inference batch size
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Ensure inputs are torch tensors
    if not torch.is_tensor(X_text):
        X_text = torch.tensor(X_text, dtype=torch.float32)
    if not torch.is_tensor(X_struct):
        X_struct = torch.tensor(X_struct, dtype=torch.float32)
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true, dtype=torch.long)

    # Convert numpy image arrays to float tensors if needed
    if isinstance(X_img, np.ndarray):
        X_img = [[torch.tensor(img, dtype=torch.float32) for img in sample] for sample in X_img]

    def collate_fn(batch_indices):
        """
        Assembles a padded batch from raw lists and returns:
        padded_imgs, text_batch, struct_batch, label_batch, attention_mask
        """
        batch_imgs = [X_img[i] for i in batch_indices]
        batch_text = X_text[batch_indices]
        batch_struct = X_struct[batch_indices]
        batch_labels = y_true[batch_indices]
        padded_imgs, mask = pad_image_sequences(batch_imgs)
        return padded_imgs, batch_text, batch_struct, batch_labels, mask

    indices = list(range(len(X_text)))
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    # Predict if needed
    if y_pred is None:
        print("Computing predictions in batches...")
        all_preds = []
        with torch.no_grad():
            for batch in batches:
                X_img_b, X_text_b, X_struct_b, _, mask_b = collate_fn(batch)
                X_img_b = X_img_b.to(device)
                X_text_b = X_text_b.to(device)
                X_struct_b = X_struct_b.to(device)
                mask_b = mask_b.to(device)

                logits = model(X_img_b, X_text_b, X_struct_b, image_mask=mask_b)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
        y_pred = all_preds

    # Compute and print metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name.capitalize()} Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    # Save outputs using standard saver
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
