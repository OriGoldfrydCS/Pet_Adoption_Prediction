"""
Main script to train and evaluate an attention-based multimodal model 
(CNN + MLP + Attention) on combined images (mean of the images), TF-IDF text, and structured data.

Main steps:
1. Loads and preprocesses the data.
2. Trains the model using training and validation sets.
3. Evaluates on both validation and test sets.
4. Saves model, predictions, and evaluation results.
"""

import sys
import os
from datetime import datetime
from time import time
import torch
import numpy as np

# === Add root directory to import modules ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# === Local imports ===
from models.SD_with_TD_with_ID.CNN_MLP_attention.model_trainer import train_and_evaluate
from models.SD_with_TD_with_ID.CNN_MLP_attention.evaluator import evaluate_model
from models.SD_with_TD_with_ID.data_loader_all_modality_mean_images_tfidf import load_text_image_structured_data


def main():
    print("Loading image + TF-IDF text + structured data...")

    # === Load and split the data ===
    (X_img_train, X_text_train, X_struct_train, y_train), \
    (X_img_val, X_text_val, X_struct_val, y_val), \
    (X_img_test, X_text_test, X_struct_test, y_test) = load_text_image_structured_data()

    # === Transpose image tensors to match PyTorch format: [B, C, H, W] ===
    X_img_train = np.transpose(X_img_train, (0, 3, 1, 2))
    X_img_val = np.transpose(X_img_val, (0, 3, 1, 2))
    X_img_test = np.transpose(X_img_test, (0, 3, 1, 2))

    # === Print shapes and sanity checks ===
    print(f"\nTrain shapes:")
    print(f"  Images:     {X_img_train.shape}")
    print(f"  Text:       {X_text_train.shape}")
    print(f"  Structured: {X_struct_train.shape}")
    print(f"  Labels:     {y_train.shape}")

    print("NaN in text:", np.isnan(X_text_train).any(), "| Inf:", np.isinf(X_text_train).any())
    print("NaN in struct:", np.isnan(X_struct_train).any(), "| Inf:", np.isinf(X_struct_train).any())
    print("NaN in labels:", np.isnan(y_train).any(), "| Unique labels:", np.unique(y_train))
    print("Text max/min:", np.max(X_text_train), np.min(X_text_train))
    print("Struct max/min:", np.max(X_struct_train), np.min(X_struct_train))

    # === Train model ===
    print("\nTraining Attention-MLP on image + TF-IDF text + structured features...")
    start = time()
    model, y_val_pred = train_and_evaluate(
        X_img_train, X_text_train, X_struct_train, y_train,
        X_img_val, X_text_val, X_struct_val, y_val
    )
    print(f"Training completed in {time() - start:.2f} seconds")

    # === Prepare output directory ===
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "attention_mlp_text_image_structured", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    # === Evaluate on validation set ===
    print("Evaluating validation set...")
    evaluate_model(
        model=model,
        X_img=X_img_val,
        X_text=X_text_val,
        X_struct=X_struct_val,
        y_true=y_val,
        y_pred=y_val_pred,
        name="val",
        base_output_dir=base_output_dir
    )

    # === Evaluate on test set (computes predictions internally) ===
    print("Evaluating test set...")
    evaluate_model(
        model=model,
        X_img=X_img_test,
        X_text=X_text_test,
        X_struct=X_struct_test,
        y_true=y_test,
        y_pred=None,
        name="test",
        base_output_dir=base_output_dir
    )

    print(f"\nAll results saved in: {base_output_dir}")


if __name__ == "__main__":
    main()
