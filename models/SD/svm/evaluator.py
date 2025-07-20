"""
Evaluation wrapper for SVM models

This utility:
- Runs prediction on a given dataset (validation or test)
- Computes and prints basic metrics (accuracy, macro-F1)
- Delegates all saving (metrics, plots, model, etc.) to `save_run_outputs`
"""

import os
from models.SD.svm.result_saver import save_run_outputs
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X, y_true, name, base_output_dir):
    """
    Evaluates a trained model and saves outputs.

    Parameters:
    - model: Trained scikit-learn model (must support `.predict()`)
    - X: Feature matrix to evaluate on
    - y_true: Ground truth labels
    - name (str): Split name for reference ('val' and 'test')
    - base_output_dir (str): Where results are saved

    Returns:
    - None (prints metrics and saves outputs)
    """
    # Generate predictions
    y_pred = model.predict(X)

    # Compute and print evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name.capitalize()} Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    # Save detailed results to disk
    save_run_outputs(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        name=name,
        base_output_dir=base_output_dir,
        use_timestamp_subfolder=False
    )
