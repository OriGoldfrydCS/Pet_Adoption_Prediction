"""
Multimodal classification model: CNN + Attention + MLP

This script loads and trains a multimodal model using:
- All available images per sample 
- TF-IDF text vectors
- Structured numeric features

Model architecture:
- CNN + Attention over image sequences
- MLP combining image, text, and structured inputs

Main steps:
1. Loads data from preprocessed pickle
2. Trains the CNN+Attn+MLP model on train/val split
3. Evaluates on validation and test sets
4. Saves model, metrics, and predictions to timestamped directory
"""

import sys
import os
from datetime import datetime
from time import time
import torch
import numpy as np

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import pipeline components
from models.SD_with_TD_with_ID.CNN_att_MLP.model_trainer import train_and_evaluate
from models.SD_with_TD_with_ID.CNN_att_MLP.evaluator import evaluate_model
from models.SD_with_TD_with_ID.data_loader_all_modality_all_images_tfidf import load_text_image_structured_data


def main():
    print("Loading image + TF-IDF text + structured data...")
    (X_img_train, X_text_train, X_struct_train, y_train), \
    (X_img_val, X_text_val, X_struct_val, y_val), \
    (X_img_test, X_text_test, X_struct_test, y_test) = load_text_image_structured_data()

    # Print example dimensions for sanity check
    print(f"\nTrain sample example:")
    print(f"  # images:     {len(X_img_train[0])}")
    print(f"  image shape:  {X_img_train[0][0].shape}")
    print(f"  text vector:  {X_text_train[0].shape}")
    print(f"  struct vector:{X_struct_train[0].shape}")
    
    print(f"\nDataset sizes: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    print(f"Text dim: {X_text_train.shape[1]}, Struct dim: {X_struct_train.shape[1]}")

    # Train model
    print("\nTraining CNN + Attention + MLP on images + TF-IDF + structured features...")
    start = time()
    model, y_val_pred = train_and_evaluate(
        X_img_train, X_text_train, X_struct_train, y_train,
        X_img_val, X_text_val, X_struct_val, y_val
    )
    print(f"Training completed in {time() - start:.2f} seconds")

    # Prepare output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "cnn_att_mlp", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    # Evaluate on validation set
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

    # Evaluate on test set (predictions computed inside)
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
