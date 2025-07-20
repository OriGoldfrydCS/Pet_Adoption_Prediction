"""
Train and evaluate a logistic regression model using BERT CLS embeddings and structured data.

- Uses a modular pipeline: data loading, training, evaluation, and saving results
- Outputs include predictions, performance metrics, and visualizations for both validation and test sets
- Results are saved under a timestamped directory

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

from models.TD_AND_TD_with_SD.logistic_text.model_trainer import train_and_evaluate
from models.TD_AND_TD_with_SD.logistic_text.evaluator import evaluate_model
from models.TD_AND_TD_with_SD.data_loader_SD_with_TD_Bert import load_train_val_test_with_bert_and_structured  # <-- updated loader

def main():
    print("Loading data (BERT CLS + structured)...")
    data = load_train_val_test_with_bert_and_structured()

    X_train, X_val, X_test = data["X_cls_combined"]
    y_train, y_val, y_test = data["y"]

    print("Data loaded.")
    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

    print("Training model...")
    start = time()
    model, y_val_pred = train_and_evaluate(X_train, y_train, X_val)
    print(f"Training completed in {time() - start:.2f} seconds")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "logistic_bert_cls_structured", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    print("Saving validation results...")
    evaluate_model(model, X_val, y_val, y_val_pred, name="val", base_output_dir=base_output_dir)

    print("Saving test results...")
    y_test_pred = model.predict(X_test)
    evaluate_model(model, X_test, y_test, y_test_pred, name="test", base_output_dir=base_output_dir)

    print(f"All results saved in: {base_output_dir}")

if __name__ == "__main__":
    main()
