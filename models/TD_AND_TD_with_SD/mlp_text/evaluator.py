"""
Evaluation wrapper for MLP model with BERT embeddings.

- Computes accuracy and macro F1
- Prints summary metrics
- Saves predictions, reports, metrics, and plots via shared result_saver utility
"""

from models.TD_AND_TD_with_SD.mlp_text.result_saver import save_run_outputs
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X, y_true, y_pred, name, base_output_dir):
    """
    Evaluates a PyTorch model and saves the results.
    Compatible with MLP, CNN, BiLSTM, and other architectures.

    Args:
        - model: Trained PyTorch model.
        - X: Input features (numpy array or tuple for multimodal input).
        - y_true: True labels.
        - y_pred: Predicted labels.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name.capitalize()} Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    # Save everything via shared result saver
    save_run_outputs(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        name=name,
        base_output_dir=base_output_dir,
        use_timestamp_subfolder=False
    )
