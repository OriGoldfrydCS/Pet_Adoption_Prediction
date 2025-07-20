"""
Evaluation wrapper for logistic regression model with BERT features (and optionally structured data).

Calls the shared output-saving utility without using timestamped subfolders.
Saves:
- Model
- Metrics
- Confusion matrix and curves
"""

from models.TD_AND_TD_with_SD.logistic_text.result_saver import save_run_outputs

def evaluate_model(model, X, y_true, y_pred, name, base_output_dir):
    save_run_outputs(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        name=name,
        base_output_dir=base_output_dir,
        use_timestamp_subfolder=False
    )
