"""
Training module for K-Nearest Neighbors (KNN) on structured data.

- Loads preprocessed train/val/test splits from split directory.
- Trains a KNN classifier using the specified number of neighbors (k).
- Returns the trained model and the data splits for evaluation or saving.
"""

from sklearn.neighbors import KNeighborsClassifier
from models.SD.data_loader_SD import load_train_val_test

def train_knn(k=1):
    """
    Trains a KNN classifier on structured features.

    Parameters:
    - k (int): Number of neighbors to use.

    Returns:
    - model (KNeighborsClassifier): Trained KNN model.
    - (X_train, y_train): Training set.
    - (X_val, y_val): Validation set.
    - (X_test, y_test): Test set.
    """
    # Load structured features for train/val/test
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test()

    # Initialize KNN model with the given number of neighbors
    model = KNeighborsClassifier(n_neighbors=k)

    # Fit model to training data
    model.fit(X_train, y_train)

    return model, (X_train, y_train), (X_val, y_val), (X_test, y_test)
