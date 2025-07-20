"""
Run script for training and evaluating a BiLSTM model using BERT token embeddings and structured data.
The model supports both multimodal input (text + structured) and text-only input, depending on the data loading strategy.
"""

import sys
import os
from datetime import datetime
from time import time
import torch
import numpy as np

# Add project root to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.TD_AND_TD_with_SD.BiLSTM_text.model_trainer import train_and_evaluate
from models.TD_AND_TD_with_SD.BiLSTM_text.evaluator import evaluate_model
from models.TD_AND_TD_with_SD.data_loader_SD_with_TD_Bert import load_train_val_test_with_bert_and_structured

def main():
    print("Loading token and structured data...")
    data = load_train_val_test_with_bert_and_structured()

    # Unpack data
    X_train_tokens, X_val_tokens, X_test_tokens = data["X_token_desc"]
    X_train_struct, X_val_struct, X_test_struct = data["X_cls_combined"]
    y_train, y_val, y_test = data["y"]

    print(f"Token shape: {X_train_tokens.shape}")             # [batch, 128, 768]
    print(f"Structured+CLS shape: {X_train_struct.shape}")    # [batch, ~1550]

    # Train model
    print("Training BiLSTM on token + structured features...")
    start = time()
    model, y_val_pred = train_and_evaluate(
        X_train_tokens, X_train_struct,
        y_train,
        X_val_tokens, X_val_struct,
        y_val
    )
    print(f"Training completed in {time() - start:.2f} seconds")

    # Create output folder
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "bilstm_bert_token_all_features", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    # Evaluate on validation set
    print("Evaluating validation set...")
    evaluate_model(
        model=model,
        X_img=X_val_tokens,
        X_text=X_val_struct,
        X_struct=y_val,
        y_true=y_val,
        y_pred=y_val_pred,
        name="val",
        base_output_dir=base_output_dir
    )

    # Evaluate on test set
    print("Evaluating test set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    X_test_tokens_tensor = torch.tensor(X_test_tokens, dtype=torch.float32).to(device)
    X_test_struct_tensor = torch.tensor(X_test_struct, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_test_tokens_tensor, X_test_struct_tensor)
        y_test_pred = torch.argmax(logits, dim=1).cpu().numpy()

    evaluate_model(
        model=model,
        X_img=X_test_tokens,
        X_text=X_test_struct,
        X_struct=y_test,
        y_true=y_test,
        y_pred=y_test_pred,
        name="test",
        base_output_dir=base_output_dir
    )

    print(f"All results saved in: {base_output_dir}")

if __name__ == "__main__":
    main()
