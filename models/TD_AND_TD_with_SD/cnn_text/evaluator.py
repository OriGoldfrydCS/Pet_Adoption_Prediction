"""
Evaluation wrapper for CNN model on BERT token embeddings + structured data.

This module:
- Computes accuracy and F1 scores
- Prints results
- Delegates saving to `save_run_outputs`, including model weights, metrics, and plots
- Supports models with two inputs: token embeddings and structured features
"""

from models.TD_AND_TD_with_SD.cnn_text.result_saver import save_run_outputs
from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(model, X_tokens, X_structured, y_true, y_pred, name, base_output_dir):
    """
    Evaluates a PyTorch classification model and saves outputs.

    Args:
        model (nn.Module): Trained PyTorch model
        X_tokens (np.array or Tensor): Token-based input (e.g., BERT/CLIP embeddings)
        X_structured (np.array or Tensor): Structured/tabular input features
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        name (str): Prefix name for saved files
        base_output_dir (str): Base directory to save evaluation results
    """
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name.capitalize()} Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    # Save results using helper function
    save_run_outputs(
        model=model,
        X=(X_tokens, X_structured),
        y_true=y_true,
        y_pred=y_pred,
        name=name,
        base_output_dir=base_output_dir,
        use_timestamp_subfolder=False
    )
