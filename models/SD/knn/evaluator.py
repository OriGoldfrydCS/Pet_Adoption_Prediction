"""
Wrapper utility to evaluate a model and delegate saving to the result_saver.

This script:
- Extracts model parameters if available
- Passes everything to the KNN `save_run_outputs` function
- Ensures all results are saved
"""

from models.SD.knn.result_saver import save_run_outputs

def evaluate_model(model, X, y_true, y_pred, name, base_output_dir):
    """
    Evaluates a trained model by passing it to the saving utility.

    Parameters:
    - model: Trained scikit-learn-style model (should implement get_params)
    - X: Input features (used for predict_proba, if supported)
    - y_true: Ground truth labels
    - y_pred: Model predictions
    - name (str): Tag for this evaluation ('val', 'test')

    Returns:
    - None
    """

    # Try to extract model parameters if available
    params = None
    if hasattr(model, "get_params"):
        params = model.get_params()

    # Save everything using shared saving function
    save_run_outputs(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        name=name,
        base_output_dir=base_output_dir,
        use_timestamp_subfolder=False,
        params=params  # this is assumed to be handled by save_run_outputs
    )
