"""
Main script for training and evaluating a CNN + Logistic Regression model
on image data combined with TF-IDF text features.

- Loads data using a custom data loader (images + TF-IDF text).
- Trains a CNN on image features and combines with logistic regression on text.
- Evaluates performance on validation and test sets.
- Saves results and predictions under timestamped output directory.
"""

import sys
import os
from datetime import datetime
from time import time
import torch
import numpy as np

# Add project root to sys.path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import training and evaluation functions
from models.ID_with_TD.CNN_LogReg.model_trainer import train_cnn_logreg
from models.ID_with_TD.CNN_LogReg.evaluator import evaluate_model
from models.ID_with_TD.data_loader_TD_with_ID_tfidf import load_text_image_data


def main():
    print("Loading image + TF-IDF text data...")
    (X_img_train, X_text_train, y_train), \
    (X_img_val, X_text_val, y_val), \
    (X_img_test, X_text_test, y_test) = load_text_image_data()

    # CNN expects images in shape [B, C, H, W]
    X_img_train = np.transpose(X_img_train, (0, 3, 1, 2))
    X_img_val = np.transpose(X_img_val, (0, 3, 1, 2))
    X_img_test = np.transpose(X_img_test, (0, 3, 1, 2))

    print(f"\nTrain shapes:")
    print(f"  Images: {X_img_train.shape}")
    print(f"  Text:   {X_text_train.shape}")
    print(f"  Labels: {y_train.shape}")

    # Train the model
    print("\nTraining CNN + LogReg on image + TF-IDF text...")
    start = time()
    model, acc, f1, y_val_pred = train_cnn_logreg(
        X_img_train, X_text_train, y_train,
        X_img_val, X_text_val, y_val,
        tfidf_dim=3000  # must match TF-IDF vectorizer
    )
    print(f"Training completed in {time() - start:.2f} seconds")

    # Create timestamped output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "cnn_LogReg_text_image", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    # Evaluate on validation set using model + val predictions
    print("Evaluating validation set...")
    evaluate_model(
        model=model,
        X_img=X_img_val,
        X_text=X_text_val,
        y_true=y_val,
        y_pred=y_val_pred,  # predictions already computed during training
        name="val",
        base_output_dir=base_output_dir
    )

    # Evaluate on test set (model will predict inside evaluate_model)
    print("Evaluating test set...")
    evaluate_model(
        model=model,
        X_img=X_img_test,
        X_text=X_text_test,
        y_true=y_test,
        y_pred=None,  # predictions will be computed here
        name="test",
        base_output_dir=base_output_dir
    )

    print(f"\nAll results saved in: {base_output_dir}")


if __name__ == "__main__":
    main()
