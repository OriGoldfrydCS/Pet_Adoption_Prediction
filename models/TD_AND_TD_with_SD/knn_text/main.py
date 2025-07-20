"""
Main script to train and evaluate a KNN model on text (Bert) + structured data.
The model supports both multimodal input (text + structured) and text-only input, depending on the data loading strategy.

Pipeline:
1. Loads BERT CLS embeddings + structured data using the dedicated loader.
2. Trains a KNN classifier on the training set and evaluates on validation.
3. Saves predictions, metrics, and model artifacts to disk.
4. Evaluates the trained model on the test set.

Directories:
- Output is saved under: performance_and_models/knn_bert_cls/YYYYMMDD_HHMMSS/ (not in the git repo)
"""

import sys
import os
from datetime import datetime
from time import time

# Add project root to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.TD_AND_TD_with_SD.knn_text.model_trainer import train_and_evaluate
from models.TD_AND_TD_with_SD.knn_text.evaluator import evaluate_model
from models.TD_AND_TD_with_SD.data_loader_TD_Bert import load_train_val_test_with_bert  # Loader with BERT CLS + structured


def main():
    print("Loading data (BERT CLS + structured)...")
    data = load_train_val_test_with_bert()

    # Unpack inputs and labels
    X_train, X_val, X_test = data["X_cls_combined"]
    y_train, y_val, y_test = data["y"]

    print("Data loaded.")
    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

    # Train KNN and evaluate on validation set
    print("Training KNN model...")
    start = time()
    model, y_val_pred = train_and_evaluate(X_train, y_train, X_val)
    print(f"Training completed in {time() - start:.2f} seconds")

    # Prepare output directory with timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "knn_bert_cls", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    # Save validation performance
    print("Saving validation results...")
    evaluate_model(model, X_val, y_val, y_val_pred, name="val", base_output_dir=base_output_dir)

    # Evaluate and save test set performance
    print("Evaluating test set...")
    y_test_pred = model.predict(X_test)
    evaluate_model(model, X_test, y_test, y_test_pred, name="test", base_output_dir=base_output_dir)

    print(f"All results saved in: {base_output_dir}")


if __name__ == "__main__":
    main()
