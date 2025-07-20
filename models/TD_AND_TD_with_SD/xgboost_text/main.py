"""
Train and evaluate an XGBoost classifier using TF-IDF features, with or without structured data.

Pipeline includes:
- Loading preprocessed text and (optional) structured data
- Training an XGBoost model
- Evaluating on validation and test sets
- Saving results (metrics, plots, etc.) to a timestamped output directory

Note:
The model supports both multimodal input (text + structured) and text-only input,
depending on the data loading strategy.
"""

import sys
import os
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.TD_AND_TD_with_SD.xgboost_text.model_trainer import train_and_evaluate
from models.TD_AND_TD_with_SD.xgboost_text.evaluator import evaluate_model
from models.TD_AND_TD_with_SD.data_loader_SD_with_TD_tfidf import load_train_val_test_with_text

def main():
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test_with_text()

    # Train model
    model, metrics = train_and_evaluate(X_train, y_train, X_val, y_val)

    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "xgboost_text", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    # Evaluate
    evaluate_model(model, X_val, y_val, name="val", base_output_dir=base_output_dir)
    evaluate_model(model, X_test, y_test, name="test", base_output_dir=base_output_dir)

if __name__ == "__main__":
    main()
