"""
Main script for training and evaluating an SVM model on structured data.

- Loads preprocessed train/val/test splits.
- Trains an SVM classifier using scikit-learn.
- Predicts on validation and test sets.
- Saves all results (metrics, plots, model, etc.) to a single shared timestamped folder.

Output is saved under:
    performance_and_models/svm/{timestamp} (not in the git repo)
"""

import sys
import os
from datetime import datetime

# Add project root to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.SD.svm.model_trainer import train_svm
from models.SD.svm.result_saver import save_run_outputs

def main():
    """
    Main pipeline:
    - Train the SVM model
    - Generate predictions on validation and test sets
    - Save results
    """
    # Train model and get all dataset splits
    model, (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_svm()

    # Predict on validation and test sets
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Create shared output directory with timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "svm", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    # Save all results (metrics, confusion matrix, model, etc.)
    save_run_outputs(model, X_val, y_val, y_val_pred, name="val", base_output_dir=base_output_dir, use_timestamp_subfolder=False)
    save_run_outputs(model, X_test, y_test, y_test_pred, name="test", base_output_dir=base_output_dir, use_timestamp_subfolder=False)

if __name__ == "__main__":
    main()
