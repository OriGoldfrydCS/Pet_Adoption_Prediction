"""
This evaluation script supports both:
1. ResNet models trained on **image data only**
2. ResNet models trained on **image + structured data**

It calculates standard evaluation metrics (accuracy, macro F1),
prints them to the console, and delegates saving all outputs
to the `save_run_outputs` utility function.
"""

from models.image_models.ResNet18_images.result_saver import save_run_outputs
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
import torch
import torch.nn.functional as F
import numpy as np

def evaluate_model(model, X, y_true, y_pred, name, base_output_dir):
    """
    Evaluate a trained model and save results.

    Parameters:
    - model: Trained PyTorch model
    - X: Input data
    - y_true, y_pred: True and predicted labels
    - name: Output name (e.g., 'test')
    - base_output_dir: Directory to save results

    Saves:
    - Accuracy, F1, log loss (if possible)
    - Confusion matrix, classification report
    - Predictions, true labels, model weights
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate accuracy and F1
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name.capitalize()} Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    # Optional: log loss calculation
    try:
        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            logits = model(X.to(device))
            probs = F.softmax(logits, dim=1).cpu().detach().numpy().astype(np.float32)

        logloss = log_loss(y_true, probs)
        print(f"{name.capitalize()} Log loss: {logloss:.4f}")
    except Exception as e:
        print(f"[WARNING] Log loss skipped due to error: {e}")

    # Save all outputs
    save_run_outputs(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        name=name,
        base_output_dir=base_output_dir,
        use_timestamp_subfolder=False
    )
