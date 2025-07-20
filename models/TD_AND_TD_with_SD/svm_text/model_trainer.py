"""
Train and evaluate an SVM classifier on text and/or structured data.

Includes:
- Model setup with class balancing and probability outputs
- Evaluation using accuracy and macro F1 on validation set

Note:
The model supports both multimodal input (text + structured) and text-only input,
depending on the data loading strategy.
"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

def train_and_evaluate(X_train, y_train, X_val, y_val):
    """
    Trains an SVM classifier and evaluates it on validation data.

    Args:
        - X_train (np.array): Training features.
        - y_train (np.array): Training labels.
        - X_val (np.array): Validation features.
        - y_val (np.array): Validation labels.

    Returns:
        - model: Trained SVM model.
        - dict: Dictionary with accuracy and macro F1 scores.
    """
    model = SVC(
        probability=True,
        kernel="rbf",
        C=10.0,
        gamma=0.1,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_val)
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1_macro": f1_score(y_val, y_pred, average="macro")
    }

    return model, metrics
