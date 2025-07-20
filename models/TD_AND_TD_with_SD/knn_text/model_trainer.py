"""
Trains and evaluates a KNN classifier using scikit-learn for the model with bert embeddings only.

Parameters:
- X_train (np.ndarray): Feature matrix for training (can include text and structured features).
- y_train (np.ndarray): Labels for training data.
- X_val (np.ndarray): Feature matrix for validation.

Returns:
- model (KNeighborsClassifier): Trained KNN model instance.
- y_val_pred (np.ndarray): Predicted labels for the validation set.
"""

from sklearn.neighbors import KNeighborsClassifier

def train_and_evaluate(X_train, y_train, X_val):
    # Define the KNN model with k=5 neighbors
    model = KNeighborsClassifier(n_neighbors=5)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on validation set
    y_val_pred = model.predict(X_val)

    return model, y_val_pred
