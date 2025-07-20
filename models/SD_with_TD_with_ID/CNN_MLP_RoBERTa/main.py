"""
Main Script: Train and Evaluate Attention-Based Multimodal Model (Images (with 5 images) + RoBERTa + Structured)

This script loads all modalities (image, RoBERTa embeddings, structured),
trains the attention-based CNN + MLP model, evaluates on validation and test sets,
and saves all outputs to a timestamped folder.
"""

import sys
import os
from datetime import datetime
from time import time
import torch
import numpy as np

# Add root to Python path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# === Import core components ===
from models.SD_with_TD_with_ID.CNN_MLP_RoBERTa.model_trainer import train_and_evaluate
from models.SD_with_TD_with_ID.CNN_MLP_RoBERTa.evaluator import evaluate_model
from models.SD_with_TD_with_ID.data_loader_all_modality_5_images_RoBERTa import load_text_image_structured_data

def main():
    # === Load dataset with 5 images, RoBERTa vectors, and structured features ===
    print("Loading image + RoBERTa text + structured data...")
    (X_img_train, X_text_train, X_struct_train, y_train), \
    (X_img_val, X_text_val, X_struct_val, y_val), \
    (X_img_test, X_text_test, X_struct_test, y_test) = load_text_image_structured_data()

    # === Check for image shape mismatches (debug) ===
    for i, img in enumerate(X_img_train):
        if img.shape != X_img_train[0].shape:
            print(f"[Mismatch] Sample {i} shape: {img.shape} vs expected {X_img_train[0].shape}")

    # === Stack and transpose image arrays to [B, C, H, W] format ===
    X_img_train = np.stack(X_img_train, axis=0).transpose(0, 1, 4, 2, 3)
    X_img_val = np.stack(X_img_val, axis=0).transpose(0, 1, 4, 2, 3)
    X_img_test = np.stack(X_img_test, axis=0).transpose(0, 1, 4, 2, 3)

    # === Print data shape and sanity checks ===
    print(f"\nTrain shapes:")
    print(f"  Images:     {X_img_train.shape}")
    print(f"  RoBERTa:    {X_text_train.shape}")
    print(f"  Structured: {X_struct_train.shape}")
    print(f"  Labels:     {y_train.shape}")
    print("NaN in RoBERTa text:", np.isnan(X_text_train).any(), "| Inf:", np.isinf(X_text_train).any())
    print("NaN in structured:", np.isnan(X_struct_train).any(), "| Inf:", np.isinf(X_struct_train).any())
    print("NaN in labels:", np.isnan(y_train).any(), "| Unique labels:", np.unique(y_train))
    print("RoBERTa max/min:", np.max(X_text_train), np.min(X_text_train))
    print("Structured max/min:", np.max(X_struct_train), np.min(X_struct_train))

    # === Train model ===
    print("\nTraining Attention-MLP on image + RoBERTa text + structured features...")
    start = time()
    model, y_val_pred = train_and_evaluate(
        X_img_train, X_text_train, X_struct_train, y_train,
        X_img_val, X_text_val, X_struct_val, y_val
    )
    print(f"Training completed in {time() - start:.2f} seconds")

    # === Create output directory ===
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "attention_mlp_roberta_image_structured", run_id)
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

    # === Evaluate on test set ===
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
