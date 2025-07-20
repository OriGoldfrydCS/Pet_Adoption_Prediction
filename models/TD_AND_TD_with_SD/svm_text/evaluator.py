"""
Evaluation wrapper for SVM model using TF-IDF or TF-IDF + structured data.

- Computes accuracy and macro F1
- Prints summary metrics
- Saves predictions, metrics, reports, and plots to the result_saver
"""

import os
from models.TD_AND_TD_with_SD.svm_text.result_saver import save_run_outputs
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X, y_true, name, base_output_dir):
    """
    Evaluate a scikit-learn SVM model with TF-IDF or TF-IDF + structured data.

    Args:
        - model: Trained scikit-learn model - SVM.
        - X: Input features (TF-IDF or TF-IDF + structured).
        - y_true: Ground truth labels.
    """
    # Make predictions
    y_pred = model.predict(X)

    # Compute simple metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name.capitalize()} Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

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
