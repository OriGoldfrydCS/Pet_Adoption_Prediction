"""
Train and evaluate an SVM classifier using TF-IDF features combined with structured data.

Pipeline includes:
- Loading preprocessed text (TF-IDF) and structured data
- Training an SVM classifier
- Evaluating on validation and test sets
- Saving results (metrics, predictions, plots) to a timestamped output directory

Note:
The model supports both multimodal input (text + structured) and text-only input,
depending on the data loading strategy.
"""

import sys
import os
from datetime import datetime
from time import time

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.TD_AND_TD_with_SD.svm_text.model_trainer import train_and_evaluate
from models.TD_AND_TD_with_SD.svm_text.evaluator import evaluate_model
from models.TD_AND_TD_with_SD.data_loader_SD_with_TD_tfidf import load_train_val_test_with_text  # <-- TF-IDF + structured loader

def main():
    print("Loading data (TF-IDF + structured)...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test_with_text()

    print("Data loaded.")
    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

    print("Training SVM model...")
    start = time()
    model, metrics = train_and_evaluate(X_train, y_train, X_val, y_val)
    print(f"Training completed in {time() - start:.2f} seconds")
    print("Validation metrics:", metrics)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "svm_tf_idf_with_structured", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    print("Saving validation results...")
    evaluate_model(model, X_val, y_val, name="val", base_output_dir=base_output_dir)

    print("Saving test results...")
    evaluate_model(model, X_test, y_test, name="test", base_output_dir=base_output_dir)

    print(f"All results saved in: {base_output_dir}")

if __name__ == "__main__":
    main()
