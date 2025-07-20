"""
Evaluation wrapper for KNN model using BERT embeddings.
Handles:
- Extracting model parameters
- Saving predictions, metrics, and visualizations to a flat output directory
"""

from models.TD_AND_TD_with_SD.knn_text.result_saver import save_run_outputs

def evaluate_model(model, X, y_true, y_pred, name, base_output_dir):
    # Try to extract model parameters if available
    params = None
    if hasattr(model, "get_params"):
        params = model.get_params()

    # Save all outputs to a shared folder without nested timestamp
    save_run_outputs(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        name=name,
        base_output_dir=base_output_dir,
        use_timestamp_subfolder=False
    )
