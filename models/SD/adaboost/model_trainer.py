"""
Training module for AdaBoost on structured data.

- Loads preprocessed train/val/test sets using the shared data loader.
- Trains an AdaBoostClassifier from scikit-learn.
- Returns the trained model and the data splits for downstream evaluation.
"""

from sklearn.ensemble import AdaBoostClassifier
from models.SD.data_loader_SD import load_train_val_test

def train_adaboost(n_estimators=300, learning_rate=1.0):
    """
    Trains an AdaBoost classifier on structured data.

    Parameters:
    - n_estimators (int): Number of boosting rounds. Default is 300.
    - learning_rate (float): Weight applied to each classifier. Default is 1.0.

    Returns:
    - model (AdaBoostClassifier): The trained AdaBoost model.
    - (X_train, y_train): Training set.
    - (X_val, y_val): Validation set.
    - (X_test, y_test): Test set.
    """
    # Load train/val/test splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test()

    # Initialize AdaBoost classifier
    model = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )

    # Fit model on training data
    model.fit(X_train, y_train)

    return model, (X_train, y_train), (X_val, y_val), (X_test, y_test)
