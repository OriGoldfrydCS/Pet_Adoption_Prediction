"""
Train and evaluate an XGBoost classifier on TF-IDF features (optionally combined with structured data).

- Uses tuned hyperparameters for multi-class classification
- Returns validation accuracy and macro F1 score

Note:
The model supports both multimodal input (text + structured) and text-only input,
depending on the data loading strategy.
"""

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_and_evaluate(X_train, y_train, X_val, y_val):
    """
    Train an XGBoost classifier and evaluate it on validation data.

    Args:
        - X_train (np.array): Training features
        - y_train (np.array): Training labels
        - X_val (np.array): Validation features
        - y_val (np.array): Validation labels

    Returns:
        - model (XGBClassifier): Trained XGBoost model
        - dict: Evaluation metrics (accuracy and macro F1)
    """
    # Initialize XGBoost model with tuned hyperparameters
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        learning_rate=0.1,
        max_depth=4,
        n_estimators=500,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.3,
        min_child_weight=2,
        random_state=42,
        verbosity=0,
        use_label_encoder=False
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_val)
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1_macro": f1_score(y_val, y_pred, average="macro")
    }

    return model, metrics
