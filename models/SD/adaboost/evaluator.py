"""
Utility function for model evaluation.
- Computes and prints accuracy, macro F1, and a full classification report.
"""

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X, y_true, name="val", save_dir=None):
    """
    Evaluates a trained model on a given dataset and prints basic metrics.

    Parameters:
    - model: Trained scikit-learn-style model with a `.predict()` method.
    - X: Feature matrix.
    - y_true: Ground truth labels.
    - name (str): Identifier for the dataset split.
    
    Returns:
    - None (prints metrics).
    """

    # Generate predictions
    y_pred = model.predict(X)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    # Print results to console
    print(f"\n--- Evaluation on {name} set ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Optional: Save confusion matrix plot
    if save_dir:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name.capitalize()} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{name}_confusion_matrix.png"))
        plt.close()
