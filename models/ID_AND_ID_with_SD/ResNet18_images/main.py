"""
This script contains two runnable sections:

1. A ResNet18 model trained **on images only**
2. A ResNet18 model trained **on images + structured data**

To run one of them, simply comment out the other.

Each script:
- Loads and preprocesses the relevant data
- Trains a ResNet18 model
- Evaluates performance on validation and test sets
- Saves results in `performance_and_models/`
"""

# This script trains and evaluates a ResNet18 model on image data only. It depends on:
# - `load_image_data()` from `data_loder_ID.py`
# - `train_and_evaluate()` from `model_trainer.py`
# - `evaluate_model()` from `evaluator.py`

import sys
import os
from datetime import datetime
from time import time
import torch
from torch.utils.data import DataLoader

# Add root path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.ID_AND_ID_with_SD.ResNet18_images.model_trainer import train_and_evaluate
from models.ID_AND_ID_with_SD.ResNet18_images.evaluator import evaluate_model
from models.ID_AND_ID_with_SD.data_loder_ID import load_image_data

def main():
    """
    Main function to train and evaluate a ResNet18 model on image-only data.
    Steps:
    - Load train/val/test image datasets
    - Train the model using training and validation sets
    - Evaluate model on validation and test sets
    - Save evaluation results (metrics, confusion matrix, report, model weights)
    """
    print("Loading data...")
    train_loader, val_loader, test_loader, all_images, all_labels, val_idx, test_idx = load_image_data()
    print("Data loaded.")

    print("Training model...")
    start = time()
    model = train_and_evaluate(train_loader, val_loader)
    print(f"Training completed in {time() - start:.2f} seconds")

    # Create output directory for saving model and results
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "ResNet50_images", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    # Set device and prepare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # ============================
    # VALIDATION PHASE
    # ============================
    print("Evaluating validation set...")
    y_val_true = []
    y_val_pred = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_val_pred.extend(preds)
            y_val_true.extend(yb.numpy())

    # Convert all val images to tensor for evaluation function
    X_val_tensor = torch.tensor(all_images[val_idx].astype("float32")).permute(0, 3, 1, 2).to(device)
    print("Evaluating validation predictions...")
    evaluate_model(
        model=model,
        X=X_val_tensor,
        y_true=y_val_true,
        y_pred=y_val_pred,
        name="val",
        base_output_dir=base_output_dir
    )

    # ============================
    # TEST PHASE
    # ============================
    print("Evaluating test set...")
    model_cpu = model.to("cpu")
    model_cpu.eval()

    y_test_true = []
    y_test_pred = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.cpu()
            outputs = model_cpu(xb)
            preds = torch.argmax(outputs, dim=1).numpy()
            y_test_pred.extend(preds)
            y_test_true.extend(yb.numpy())

    # Convert all test images to tensor for evaluation function
    X_test_tensor = torch.tensor(all_images[test_idx].astype("float32")).permute(0, 3, 1, 2).cpu()
    print("Evaluating test predictions...")
    evaluate_model(
        model=model_cpu,
        X=X_test_tensor,
        y_true=y_test_true,
        y_pred=y_test_pred,
        name="test",
        base_output_dir=base_output_dir
    )

    print(f"All results saved in: {base_output_dir}")

if __name__ == "__main__":
    main()




# This script trains and evaluates a ResNet18 model using both image data and structured features. It depends on:
# - `load_image_and_structured_data()` from `data_loder_SD_and_ID.py`
# - `train_and_evaluate()` from `model_trainer.py`
# - `evaluate_model()` from `evaluator.py`

import sys
import os
from datetime import datetime
from time import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add root path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.ID_AND_ID_with_SD.ResNet18_images.model_trainer import train_and_evaluate
from models.ID_AND_ID_with_SD.ResNet18_images.evaluator import evaluate_model
from models.ID_AND_ID_with_SD.data_loder_SD_and_ID import load_image_and_structured_data

def main():
    """
    Main function to train and evaluate a ResNet18 model using both images and structured data.
    Steps:
    - Load training/validation/test datasets (image + structured features)
    - Train the model using both modalities
    - Evaluate on validation and test sets
    - Save results: metrics, plots, logs, and model outputs
    """
    print("Loading image + structured feature data...")
    data = load_image_and_structured_data()

    X_img_train, X_img_val, X_img_test = data["X_img"]
    X_struct_train, X_struct_val, X_struct_test = data["X_struct"]
    y_train, y_val, y_test = data["y"]

    print("Data loaded.")
    print(f"Train images: {X_img_train.shape}, Train features: {X_struct_train.shape}")
    print(f"Val images:   {X_img_val.shape}, Val features:   {X_struct_val.shape}")
    print(f"Test images:  {X_img_test.shape}, Test features:  {X_struct_test.shape}")

    num_structured_features = X_struct_train.shape[1]

    # ============================
    # TRAINING PHASE
    # ============================
    print("Training model...")
    start = time()
    model, y_val_pred, logs = train_and_evaluate(
        X_img_train, X_struct_train, y_train,
        X_img_val, X_struct_val, y_val,
        num_structured_features
    )
    print(f"Training completed in {time() - start:.2f} seconds")

    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "ResNet18_images_and_features", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    # Save training logs
    import json
    with open(os.path.join(base_output_dir, "training_logs.json"), "w") as f:
        json.dump(logs, f, indent=2)

    # ============================
    # VALIDATION PHASE
    # ============================
    print("Evaluating validation set...")
    evaluate_model(
        model=model,
        X=(X_img_val, X_struct_val),
        y_true=y_val,
        y_pred=y_val_pred,
        name="val",
        base_output_dir=base_output_dir
    )

    # ============================
    # TEST PHASE
    # ============================
    print("Evaluating test set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert test data to tensors
    X_img_tensor = torch.tensor(X_img_test, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    X_struct_tensor = torch.tensor(X_struct_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_img_tensor, X_struct_tensor, y_test_tensor), batch_size=64)

    y_test_pred = []
    with torch.no_grad():
        model.eval()
        for xb_img, xb_struct, _ in test_loader:
            xb_img, xb_struct = xb_img.to(device), xb_struct.to(device)
            outputs = model(xb_img, xb_struct)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_test_pred.extend(preds)

    print("Evaluating test predictions...")
    evaluate_model(
        model=model,
        X=(X_img_test, X_struct_test),
        y_true=y_test,
        y_pred=y_test_pred,
        name="test",
        base_output_dir=base_output_dir
    )

    print(f"All results saved in: {base_output_dir}")

if __name__ == "__main__":
    main()
