"""
Evaluation wrapper for BiLSTM models using token + structured input.
"""

from models.TD_AND_TD_with_SD.BiLSTM_text.result_saver import save_run_outputs
from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(model, X_tokens, X_struct, y_true, y_pred, name, base_output_dir):
    """
    Evaluates a PyTorch model on a given dataset and prints/saves the results.

    Works with models that accept two inputs: token embeddings and structured features.
    Assumes predictions have already been generated.

    Args:
        - model: Trained PyTorch model.
        - X_tokens (np.ndarray): Token-level input features.
        - X_struct (np.ndarray): Structured features (e.g., metadata).
        - y_true (np.ndarray): Ground-truth labels.
        - y_pred (np.ndarray): Predicted labels.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"{name.capitalize()} Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    save_run_outputs(
        model=model,
        X=(X_tokens, X_struct),
        y_true=y_true,
        y_pred=y_pred,
        name=name,
        base_output_dir=base_output_dir,
        use_timestamp_subfolder=False
    )
