"""
Training module for XGBoost classifier on structured data.

- Loads pre-split train/val/test datasets.
- Initializes and trains an XGBoost model with tuned hyperparameters.
- Returns the trained model and all three data splits.

Note:
- Assumes a multi-class classification problem with 4 classes.
- `softprob` objective returns probability distribution for each class.
- `use_label_encoder=False` disables legacy label encoder.
"""

from xgboost import XGBClassifier
from models.SD.data_loader_SD import load_train_val_test

def train_xgboost():
    """
    Trains an XGBoost classifier on structured features.

    Returns:
    - model (XGBClassifier): Trained model instance.
    - (X_train, y_train): Training set.
    - (X_val, y_val): Validation set.
    - (X_test, y_test): Test set.
    """
    # Load structured train/val/test datasets
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test()

    # Define the XGBoost model with tuned hyperparameters
    model = XGBClassifier(
        objective="multi:softprob",   # multi-class with probability output
        num_class=4,                  # number of target classes
        learning_rate=0.05,
        max_depth=6,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        min_child_weight=5,
        random_state=42,
        verbosity=0,
        use_label_encoder=False
    )

    # Train the model on training data
    model.fit(X_train, y_train)

    return model, (X_train, y_train), (X_val, y_val), (X_test, y_test)
