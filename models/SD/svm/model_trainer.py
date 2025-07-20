"""
Training module for Support Vector Machine (SVM) on structured data.

- Loads preprocessed train/val/test splits from split directory.
- Initializes an SVM model with predefined hyperparameters.
- Trains the model on the training set.
- Returns the trained model and all data splits for evaluation and saving.

Note:
- `probability=True` is enabled to allow use of `predict_proba` for evaluation (e.g., ROC, log loss).
- `class_weight="balanced"` is used to account for class imbalance.
"""

from sklearn.svm import SVC
from models.SD.data_loader_SD import load_train_val_test

def train_svm():
    """
    Trains an SVM classifier on structured features.

    Returns:
    - model (SVC): Trained SVM model with probability support.
    - (X_train, y_train): Training set
    - (X_val, y_val): Validation set
    - (X_test, y_test): Test set
    """
    # Load structured features for train/val/test
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test()

    # Initialize the SVM model with chosen hyperparameters
    model = SVC(
        probability=True,        # Enables predict_proba()
        kernel="rbf",            # Radial basis function kernel
        C=10.0,                  # Regularization strength
        gamma=0.1,               # Kernel coefficient
        class_weight="balanced", # Handle class imbalance
        random_state=42
    )

    # Fit model to training data
    model.fit(X_train, y_train)

    return model, (X_train, y_train), (X_val, y_val), (X_test, y_test)
