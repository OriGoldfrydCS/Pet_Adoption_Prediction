"""
Main script for training and evaluating an AdaBoost model on structured data.

- Loads preprocessed train/val/test splits from CSV files.
- Trains an AdaBoost classifier on the structured features.
- Generates predictions on the validation and test sets.
- Saves the trained model, metrics, and predictions to a timestamped output folder.
- Output is saved under: performance_and_models/adaboost/{timestamp} (not in the git repo)
"""

import sys
import os
from datetime import datetime

# Add project root to sys.path so we can import modules easily
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.SD.adaboost.model_trainer import train_adaboost
from models.SD.adaboost.result_saver import save_run_outputs

def main():
    """
    Main execution function:
    - Trains the AdaBoost model
    - Evaluates on validation and test sets
    - Saves predictions, metrics, and model to output folder
    """
    # Train model and get train/val/test splits
    model, (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_adaboost()

    # Predict on validation and test data
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Generate timestamped output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "adaboost", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    # Save all outputs to disk
    save_run_outputs(model, X_val, y_val, y_val_pred, name="val", base_output_dir=base_output_dir)
    save_run_outputs(model, X_test, y_test, y_test_pred, name="test", base_output_dir=base_output_dir)

if __name__ == "__main__":
    main()
