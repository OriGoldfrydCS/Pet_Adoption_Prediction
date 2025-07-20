"""
Main script for training and evaluating a CNN + MLP model
on image data combined with TF-IDF text features.

- Loads structured image + text dataset.
- Trains CNN for image processing and MLP for classification.
- Evaluates model on validation and test sets.
- Saves all outputs to a timestamped results folder.
"""

import sys
import os
from datetime import datetime
from time import time
import torch
import numpy as np

# Add project root to Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import training, evaluation, and data loading functions
from models.ID_with_TD.CNN_MLP.model_trainer import train_and_evaluate
from models.ID_with_TD.CNN_MLP.evaluator import evaluate_model
from models.ID_with_TD.data_loader_TD_with_ID_tfidf import load_text_image_data


def main():
    print("Loading image + TF-IDF text data...")
    (X_img_train, X_text_train, y_train), \
    (X_img_val, X_text_val, y_val), \
    (X_img_test, X_text_test, y_test) = load_text_image_data()

    # Convert images from [B, H, W, C] to [B, C, H, W] for CNN input
    X_img_train = np.transpose(X_img_train, (0, 3, 1, 2))
    X_img_val = np.transpose(X_img_val, (0, 3, 1, 2))
    X_img_test = np.transpose(X_img_test, (0, 3, 1, 2))

    print(f"\nTrain shapes:")
    print(f"  Images: {X_img_train.shape}")
    print(f"  Text:   {X_text_train.shape}")
    print(f"  Labels: {y_train.shape}")

    # Train the model
    print("\nTraining CNN + MLP on image + TF-IDF text...")
    start = time()
    model, y_val_pred = train_and_evaluate(
        X_img_train, X_text_train, y_train,
        X_img_val, X_text_val, y_val
    )
    print(f"Training completed in {time() - start:.2f} seconds")

    # Prepare output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "cnn_mlp_text_image", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    # Evaluate on validation set
    print("Evaluating validation set...")
    evaluate_model(
        model=model,
        X_img=X_img_val,
        X_text=X_text_val,
        y_true=y_val,
        y_pred=y_val_pred,  # already predicted during training
        name="val",
        base_output_dir=base_output_dir
    )

    # Evaluate on test set
    print("Evaluating test set...")
    evaluate_model(
        model=model,
        X_img=X_img_test,
        X_text=X_text_test,
        y_true=y_test,
        y_pred=None,  # will be computed inside evaluator
        name="test",
        base_output_dir=base_output_dir
    )

    print(f"\nAll results saved in: {base_output_dir}")


# Entry point
if __name__ == "__main__":
    main()
