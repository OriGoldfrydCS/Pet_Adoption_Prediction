"""
Evaluation utility for XGBoost.

This script:
- Generates predictions for a given dataset.
- Computes and prints accuracy and macro F1 score.
- Delegates saving of all outputs (metrics, plots, model, etc.) to `save_run_outputs`.
"""

import os
from models.SD.xgboost.result_saver import save_run_outputs
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X, y_true, name, base_output_dir):
    """
    Evaluates a trained model and saves outputs.

    Parameters:
    - model: Trained scikit-learn-style classifier.
    - X: Feature matrix for evaluation.
    - y_true: True labels.
    - name (str): Tag for the current evaluation ('val' and 'test').
    - base_output_dir (str): Directory where outputs are saved.

    Returns:
    - None (prints metrics and saves results).
    """
    # Predict labels
    y_pred = model.predict(X)

    # Compute and print metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name.capitalize()} Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    # Save outputs using result_saver module
    save_run_outputs(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        name=name,
        base_output_dir=base_output_dir,
        use_timestamp_subfolder=False
    )
