"""
Main script for training and evaluating a CNN + LSTM model using mean image data
and BERT embeddings as text input—either CLS token or full-sequence—based on the mode
specified in the data loader ("cls" or "all").

- Loads image data and [T, 768] BERT embeddings.
- Trains a hybrid CNN + LSTM model.
- Evaluates performance on validation and test sets.
- Saves results to a timestamped output folder.
"""


import sys
import os
from datetime import datetime
from time import time
import torch
import numpy as np

# Add project root to sys.path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import training, evaluation, and data loading
from models.ID_with_TD.CNN_LSTM.model_trainer import train_cnn_lstm
from models.ID_with_TD.CNN_LSTM.evaluator import evaluate_model
from models.ID_with_TD.data_loader_TD_with_ID_Bert import load_text_image_data


def main():
    # Detect and print device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load image and BERT sequence data (shape: [B, T, 768])
    print("Loading image + BERT (full sequence) text data...")
    (X_img_train, X_text_train, y_train), \
    (X_img_val, X_text_val, y_val), \
    (X_img_test, X_text_test, y_test) = load_text_image_data(bert_mode="all")

    # Convert images from [B, H, W, C] to [B, C, H, W]
    X_img_train = np.transpose(X_img_train, (0, 3, 1, 2))
    X_img_val = np.transpose(X_img_val, (0, 3, 1, 2))
    X_img_test = np.transpose(X_img_test, (0, 3, 1, 2))

    print(f"\nTrain shapes:")
    print(f"  Images: {X_img_train.shape}")
    print(f"  Text:   {X_text_train.shape}")  # Should be [B, T, 768]
    print(f"  Labels: {y_train.shape}")

    # Train the model
    print("\nTraining CNN + LSTM (BERT full sequence)...")
    start = time()
    model, acc, f1, y_val_pred = train_cnn_lstm(
        X_img_train, X_text_train, y_train,
        X_img_val, X_text_val, y_val,
    )
    print(f"Training completed in {time() - start:.2f} seconds")

    # Create output directory with timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "cnn_LSTM_text_image", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    # Evaluate on validation set
    print("Evaluating validation set...")
    evaluate_model(
        model=model,
        X_img=X_img_val,
        X_text_seq=X_text_val,
        y_true=y_val,
        y_pred=y_val_pred,  # predictions already computed during training
        name="val",
        base_output_dir=base_output_dir,
    )

    # Evaluate on test set
    print("Evaluating test set...")
    evaluate_model(
        model=model,
        X_img=X_img_test,
        X_text_seq=X_text_test,
        y_true=y_test,
        y_pred=None,  # predictions will be computed inside evaluator
        name="test",
        base_output_dir=base_output_dir,
    )

    print(f"\nAll results saved in: {base_output_dir}")


# Entry point
if __name__ == "__main__":
    main()
