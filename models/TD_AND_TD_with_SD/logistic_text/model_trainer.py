"""
Train a multinomial logistic regression model and evaluate on validation set.

Returns:
- Trained model
- Validation predictions
"""

from sklearn.linear_model import LogisticRegression

def train_and_evaluate(X_train, y_train, X_val):
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        C=1.0
    )

    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    return model, y_val_pred
