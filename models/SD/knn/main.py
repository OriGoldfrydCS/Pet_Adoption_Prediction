"""
Main script for training and evaluating a K-Nearest Neighbors (KNN) model on structured data.

- Loads preprocessed train/val/test splits from CSV files.
- Trains a KNN classifier with a user-defined number of neighbors (k).
- Evaluates performance on validation and test sets.
- Saves predictions, metrics, model parameters, and visualizations to a timestamped output directory.

Usage:
    python main.py [k]

If no `k` is provided via command line, the default is k=5.
"""

import sys
import os
from datetime import datetime

# Add project root to sys.path to allow relative imports from anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.SD.knn.model_trainer import train_knn
from models.SD.knn.result_saver import save_run_outputs

def main(k):
    """
    Executes the full KNN training and evaluation pipeline with the specified k.

    Steps:
    - Train KNN model with `k` neighbors
    - Predict on validation and test sets
    - Save all results (metrics, plots, model) to output directory

    Parameters:
    - k (int): Number of neighbors to use for KNN
    """
    # Train the model and get all data splits
    model, (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_knn(k)

    # Predict on validation and test sets
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Create a base directory with timestamp (one shared folder for both val and test)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "knn", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    # Save outputs: model, metrics, predictions, confusion matrix, etc.
    save_run_outputs(
        model, X_val, y_val, y_val_pred,
        name="val", base_output_dir=base_output_dir,
        use_timestamp_subfolder=False,
        X_train=X_train, y_train=y_train
    )

    save_run_outputs(
        model, X_test, y_test, y_test_pred,
        name="test", base_output_dir=base_output_dir,
        use_timestamp_subfolder=False
    )

if __name__ == "__main__":
    # Allow k to be passed from command line
    if len(sys.argv) > 1:
        k = int(sys.argv[1])
    else:
        k = 5  # Default to k=5 if not provided
    main(k)
