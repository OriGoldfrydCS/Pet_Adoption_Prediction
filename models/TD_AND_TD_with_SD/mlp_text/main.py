"""
Train and evaluate an MLP model on full/cls BERT token embeddings and structured data.

Pipeline includes:
- Loading BERT token sequences for name and description + structured data
- Flattening and concatenating inputs
- Training an MLP classifier
- Evaluating on both validation and test sets
- Saving predictions, metrics, and plots to a timestamped directory

Note:
The model supports both multimodal input (text + structured) and text-only input,
depending on the data loading strategy.
"""
import sys
import os
from datetime import datetime
from time import time
import torch
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.TD_AND_TD_with_SD.mlp_text.model_trainer import train_and_evaluate
from models.TD_AND_TD_with_SD.mlp_text.evaluator import evaluate_model
from models.TD_AND_TD_with_SD.data_loader_SD_with_TD_Bert import load_train_val_test_with_bert_and_structured

def main():
    print("Loading BERT token embeddings + structured data...")
    data = load_train_val_test_with_bert_and_structured()

    # Extract components
    desc_train, desc_val, desc_test = data["X_token_desc"]
    name_train, name_val, name_test = data["X_token_name"]
    cls_comb_train, cls_comb_val, cls_comb_test = data["X_cls_combined"]
    y_train, y_val, y_test = data["y"]

    # Flatten token embeddings: [N, 128, 768] → [N, 128*768]
    desc_train = desc_train.reshape(desc_train.shape[0], -1)
    desc_val   = desc_val.reshape(desc_val.shape[0], -1)
    desc_test  = desc_test.reshape(desc_test.shape[0], -1)

    name_train = name_train.reshape(name_train.shape[0], -1)
    name_val   = name_val.reshape(name_val.shape[0], -1)
    name_test  = name_test.reshape(name_test.shape[0], -1)

    # Extract structured part from cls_combined (last columns)
    struct_train = cls_comb_train[:, 768*2:]
    struct_val   = cls_comb_val[:, 768*2:]
    struct_test  = cls_comb_test[:, 768*2:]

    # Concatenate: full token embeddings + structured
    X_train = np.hstack([desc_train, name_train, struct_train])
    X_val   = np.hstack([desc_val,   name_val,   struct_val])
    X_test  = np.hstack([desc_test,  name_test,  struct_test])

    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    print("Training MLP on full BERT token embeddings + structured features...")
    start = time()
    model, y_val_pred = train_and_evaluate(X_train, y_train, X_val, y_val)
    print(f"Training completed in {time() - start:.2f} seconds")

    # Output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "mlp_bert_all_features", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    print("Evaluating validation set...")
    evaluate_model(model, X_val, y_val, y_val_pred, name="val", base_output_dir=base_output_dir)

    print("Evaluating test set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(X_test_tensor)
        y_test_pred = torch.argmax(logits, dim=1).cpu().numpy()

    evaluate_model(model, X_test, y_test, y_test_pred, name="test", base_output_dir=base_output_dir)

    print(f"All results saved in: {base_output_dir}")


if __name__ == "__main__":
    main()
