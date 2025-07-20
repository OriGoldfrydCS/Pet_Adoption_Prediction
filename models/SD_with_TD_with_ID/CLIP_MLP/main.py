"""
This script loads a multimodal dataset containing:
- CLIP-encoded image embeddings (averaged over 10 images per sample/or used all the 10 images per sample)
- CLIP-encoded text embeddings
- Structured numeric features

It trains an MLP classifier on all three modalities using a train/val split,
and evaluates the trained model on both validation and test sets.

All results (metrics, plots, predictions) are saved under:
`performance_and_models/clip_mlp_all_modalities_10_images/{timestamp}`

Note:
- Inputs are already preprocessed as 2D arrays: image/text (512-dim), structured (normalized).
- CLIP features are expected to come from the associated data loader script.
"""

import sys
import os
from datetime import datetime
from time import time
import torch
import numpy as np

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import model pipeline
from models.SD_with_TD_with_ID.CLIP_MLP.model_trainer import train_and_evaluate
from models.SD_with_TD_with_ID.CLIP_MLP.evaluator import evaluate_model
from models.SD_with_TD_with_ID.data_loader_all_modality_10_images_CLIP import load_text_image_structured_data


def main():
    print("Loading image + CLIP text + structured data...")
    (X_img_train, X_text_train, X_struct_train, y_train), \
    (X_img_val, X_text_val, X_struct_val, y_val), \
    (X_img_test, X_text_test, X_struct_test, y_test) = load_text_image_structured_data()

    # Check input shapes
    print(f"\nTrain shapes:")
    print(f"  CLIP Image:  {X_img_train.shape}")
    print(f"  CLIP Text:   {X_text_train.shape}")
    print(f"  Structured:  {X_struct_train.shape}")
    print(f"  Labels:      {y_train.shape}")

    # Basic sanity checks
    print("NaN in CLIP text:", np.isnan(X_text_train).any(), "| Inf:", np.isinf(X_text_train).any())
    print("NaN in structured:", np.isnan(X_struct_train).any(), "| Inf:", np.isinf(X_struct_train).any())
    print("NaN in labels:", np.isnan(y_train).any(), "| Unique labels:", np.unique(y_train))
    print("CLIP text max/min:", np.max(X_text_train), np.min(X_text_train))
    print("Structured max/min:", np.max(X_struct_train), np.min(X_struct_train))

    # Train the model
    print("\nTraining CLIP-MLP on image + text + structured features...")
    start = time()
    model, y_val_pred = train_and_evaluate(
        X_img_train, X_text_train, X_struct_train, y_train,
        X_img_val, X_text_val, X_struct_val, y_val
    )
    print(f"Training completed in {time() - start:.2f} seconds")

    # Prepare output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "clip_mlp_all_modalities_10_images", run_id)
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

    # Evaluate on test set (y_pred is computed inside evaluator)
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
