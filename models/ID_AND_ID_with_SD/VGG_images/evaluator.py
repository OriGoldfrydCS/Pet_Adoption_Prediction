"""
This evaluation script supports both:
1. VGG models trained on **image data only**
2. VGG models trained on **image + structured data**

It calculates standard evaluation metrics (accuracy, macro F1),
prints them to the console, and delegates saving all outputs
to the `save_run_outputs` utility function.
"""

from models.image_models.VGG_images.result_saver import save_run_outputs
from sklearn.metrics import accuracy_score, f1_score, log_loss
import torch
import torch.nn.functional as F
import numpy as np

def evaluate_model(model, X, y_true, y_pred, name, base_output_dir, logs=None):
    """
    Evaluates a trained model and saves evaluation results.

    Parameters:
    - model: Trained PyTorch model
    - X: Input data (Tensor or tuple, depending on model type)
    - y_true: Ground-truth labels (list or array)
    - y_pred: Predicted labels (list or array)
    - name: Identifier for the evaluation phase (e.g., 'test', 'val')
    - base_output_dir: Directory path to store evaluation results
    - logs: Optional training log data (list of dicts), passed to the saver

    Output:
    - Prints accuracy, F1, and log loss (if computed)
    - Saves model weights, predictions, metrics, and visualizations
    """
    # Ensure y_true and y_pred are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name.capitalize()} Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    # Try calculating log loss
    try:
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            logits = model(X.to(device))
            probs = F.softmax(logits, dim=1).cpu().detach().numpy().astype(np.float32)

        # Ensure y_true is int64
        if y_true.dtype != np.int64 and y_true.dtype != np.int32:
            y_true = y_true.astype(np.int64)

        logloss = log_loss(y_true, probs)
        print(f"{name.capitalize()} Log loss: {logloss:.4f}")
    except Exception as e:
        print(f"[WARNING] Log loss skipped due to error: {e}")
        logloss = None

    # Save everything
    save_run_outputs(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        name=name,
        base_output_dir=base_output_dir,
        use_timestamp_subfolder=False,
    )
