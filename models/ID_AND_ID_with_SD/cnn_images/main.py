"""
This script contains two runnable sections:

1. A CNN model trained **on images only**
2. A CNN model trained **on images + structured data**

To run one of them, simply comment out the other.

Each script:
- Loads and preprocesses the relevant data
- Trains a CNN model
- Evaluates performance on validation and test sets
- Saves results in `performance_and_models/`
"""


# This script trains and evaluates a CNN model on image data only. It depends on:
# - `load_image_data()` from `data_loder_ID.py`
# - `train_and_evaluate()` from `model_trainer.py`
# - `evaluate_model()` from `evaluator.py`

import sys
import os
from datetime import datetime
from time import time
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add root path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.ID_AND_ID_with_SD.cnn_images.model_trainer import train_and_evaluate
from models.ID_AND_ID_with_SD.cnn_images.evaluator import evaluate_model
from models.ID_AND_ID_with_SD.data_loder_ID import load_image_data  # Returns DataLoaders

def main():
    print("Loading data...")
    loaders = load_image_data()
    train_loader, val_loader, test_loader = loaders[:3]
    print("Data loaded.")

    print("Training model...")
    start = time()
    model = train_and_evaluate(train_loader, val_loader)
    print(f"Training completed in {time() - start:.2f} seconds")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "cnn_images", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # --- Evaluate validation set ---
    print("Evaluating validation set...")
    val_preds, val_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(yb.numpy())

    evaluate_model(model, None, val_labels, val_preds, name="val", base_output_dir=base_output_dir)

    # --- Evaluate test set ---
    print("Evaluating test set...")
    test_preds, test_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(yb.numpy())

    evaluate_model(model, None, test_labels, test_preds, name="test", base_output_dir=base_output_dir)

    print(f"All results saved in: {base_output_dir}")

if __name__ == "__main__":
    main()


# This script trains and evaluates a CNN model using both image data and structured features. It depends on: 
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
from sklearn.preprocessing import StandardScaler

# Add root path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.ID_AND_ID_with_SD.cnn_images.model_trainer import train_and_evaluate
from models.ID_AND_ID_with_SD.cnn_images.evaluator import evaluate_model
from models.ID_AND_ID_with_SD.data_loder_SD_and_ID import load_image_and_structured_data

def main():
    print("Loading image and structured data...")
    data = load_image_and_structured_data()
    X_train_img, X_val_img, X_test_img = data["X_img"]
    X_train_struct, X_val_struct, X_test_struct = data["X_struct"]
    y_train, y_val, y_test = data["y"]

    # Optionally reduce train set size for quick experiments
    X_train_img = X_train_img[:100]
    X_train_struct = X_train_struct[:100]
    y_train = y_train[:100]

    print(f"Train set: {len(X_train_img)} samples")
    print(f"Val set: {len(X_val_img)} samples")
    print(f"Test set: {len(X_test_img)} samples")
    print(f"Image shape: {X_train_img.shape}")
    print(f"Structured shape: {X_train_struct.shape}")

    # --- Normalize and standardize inputs ---
    X_train_img = X_train_img / 255.0
    X_val_img = X_val_img / 255.0
    X_test_img = X_test_img / 255.0

    scaler = StandardScaler()
    X_train_struct = scaler.fit_transform(X_train_struct)
    X_val_struct = scaler.transform(X_val_struct)
    X_test_struct = scaler.transform(X_test_struct)

    # --- Convert to PyTorch tensors ---
    def prepare_tensors(X_img, X_struct, y):
        if isinstance(X_img, np.ndarray):
            X_img = torch.tensor(X_img, dtype=torch.float32).permute(0, 3, 1, 2)  # NHWC → NCHW
        if isinstance(X_struct, np.ndarray):
            X_struct = torch.tensor(X_struct, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.long)
        return X_img, X_struct, y

    X_train_img, X_train_struct, y_train = prepare_tensors(X_train_img, X_train_struct, y_train)
    X_val_img,   X_val_struct,   y_val   = prepare_tensors(X_val_img,   X_val_struct,   y_val)
    X_test_img,  X_test_struct,  y_test  = prepare_tensors(X_test_img,  X_test_struct,  y_test)

    num_structured_features = X_train_struct.shape[1]

    # --- Print label distribution ---
    print("Label distribution:")
    print("Train:", np.bincount(y_train.numpy()))
    print("Val  :", np.bincount(y_val.numpy()))
    print("Test :", np.bincount(y_test.numpy()))

    print("Training CNN model with images + features...")
    start = time()
    model, y_val_pred = train_and_evaluate(
        X_train_img, X_train_struct, y_train,
        X_val_img, X_val_struct, y_val,
        num_structured_features
    )
    print(f"Training completed in {time() - start:.2f} seconds")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "cnn_images_and_features", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    print("Evaluating validation set...")
    evaluate_model(model, (X_val_img, X_val_struct), y_val, y_val_pred, name="val", base_output_dir=base_output_dir)

    # --- Evaluate test set ---
    print("Evaluating test set...")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_test_preds = []
    test_loader = DataLoader(TensorDataset(X_test_img, X_test_struct, y_test), batch_size=64)
    with torch.no_grad():
        for xb_img, xb_struct, _ in test_loader:
            xb_img = xb_img.to(device)
            xb_struct = xb_struct.to(device)
            logits = model(xb_img, xb_struct)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_test_preds.extend(preds)

    evaluate_model(model, (X_test_img, X_test_struct), y_test, y_test_preds, name="test", base_output_dir=base_output_dir)
    print(f"All results saved in: {base_output_dir}")

if __name__ == "__main__":
    main()

