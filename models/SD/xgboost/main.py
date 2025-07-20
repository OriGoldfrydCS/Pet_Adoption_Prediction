"""
Main script for training and evaluating an XGBoost model on structured data.

Pipeline overview:
- Loads preprocessed train/val/test splits.
- Trains an XGBoost classifier using predefined hyperparameters.
- Evaluates performance on both validation and test sets.
- Saves results (metrics, plots, model, etc.) in a single shared timestamped directory.

Results are stored under:
    performance_and_models/xgboost/{timestamp} (not in the git repo)
"""

import sys
import os
from datetime import datetime

# Add project root to sys.path to enable relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.SD.xgboost.model_trainer import train_xgboost
from models.SD.xgboost.evaluator import evaluate_model

def main():
    """
    Full training and evaluation pipeline for XGBoost:
    - Train on structured data
    - Evaluate on validation and test sets
    - Save everything
    """
    # Train the model and retrieve dataset splits
    model, (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_xgboost()

    # Create output directory with timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "xgboost", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    # Evaluate model and save outputs
    evaluate_model(model, X_val, y_val, name="val", base_output_dir=base_output_dir)
    evaluate_model(model, X_test, y_test, name="test", base_output_dir=base_output_dir)

if __name__ == "__main__":
    main()
