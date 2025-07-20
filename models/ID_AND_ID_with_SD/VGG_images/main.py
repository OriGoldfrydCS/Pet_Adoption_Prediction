"""
This script contains two runnable sections:

1. A VGG model trained **on images only**
2. A VGG model trained **on images + structured data**

To run one of them, simply comment out the other.

Each script:
- Loads and preprocesses the relevant data
- Trains a VGG19 model
- Evaluates performance on validation and test sets
- Saves results in `performance_and_models/`
"""

# This script trains and evaluates a VGG model on image data only. It depends on:
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

from models.ID_AND_ID_with_SD.VGG_images.model_trainer import train_and_evaluate
from models.ID_AND_ID_with_SD.VGG_images.evaluator import evaluate_model
from models.ID_AND_ID_with_SD.data_loder_ID import load_image_data

def main():
    """
    Trains and evaluates a VGG19 model using image data only.
    Loads image data, trains the model, evaluates it on validation and test sets,
    and saves the results to a timestamped folder.
    """
    print("Loading data...")
    loaders = load_image_data()
    train_loader, val_loader, test_loader = loaders[:3]
    print("Data loaded.")

    print("Training model...")
    start = time()
    model = train_and_evaluate(train_loader, val_loader)
    print(f"Training completed in {time() - start:.2f} seconds")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "vgg_images", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    print("Evaluating validation set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    all_preds_val = []
    all_labels_val = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds_val.extend(preds)
            all_labels_val.extend(yb.numpy())

    evaluate_model(model, None, all_labels_val, all_preds_val, name="val", base_output_dir=base_output_dir)

    print("Evaluating test set...")
    all_preds_test = []
    all_labels_test = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds_test.extend(preds)
            all_labels_test.extend(yb.numpy())

    evaluate_model(model, None, all_labels_test, all_preds_test, name="test", base_output_dir=base_output_dir)

    print(f"All results saved in: {base_output_dir}")

if __name__ == "__main__":
    main()



# This script trains and evaluates a VGG model using both image data and structured features. It depends on: 
# - `load_image_and_structured_data()` from `data_loder_SD_and_ID.py`
# - `train_and_evaluate()` from `model_trainer.py`
# - `evaluate_model()` from `evaluator.py`

import sys
import os
from datetime import datetime
from time import time
import numpy as np
import torch
import gc
from torch.utils.data import DataLoader, TensorDataset

# Memory cleanup
torch.cuda.empty_cache()
gc.collect()

# Add root path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.ID_AND_ID_with_SD.VGG_images.model_trainer import train_and_evaluate, SmallVGG11WithFeatures
from models.ID_AND_ID_with_SD.VGG_images.evaluator import evaluate_model
from models.ID_AND_ID_with_SD.data_loder_SD_and_ID import load_image_and_structured_data


def main():
    """
    Trains and evaluates a VGG11 model using image and structured data.
    Loads both image and structured data, trains the model, evaluates it on validation and test sets,
    and saves the results to a timestamped folder.
    """
    print("Loading image and structured data...")
    data = load_image_and_structured_data()
    (X_train_img, X_val_img, X_test_img) = data["X_img"]
    (X_train_struct, X_val_struct, X_test_struct) = data["X_struct"]
    (y_train, y_val, y_test) = data["y"]
    print("Data loaded.")
    print(f"Image shape: {X_train_img.shape}")
    print(f"Structured shape: {X_train_struct.shape}")

    print("Training VGG11 model with images + features...")
    start = time()
    model, y_val_pred = train_and_evaluate(
        X_train_img, X_train_struct, y_train,
        X_val_img, X_val_struct, y_val,
        num_structured_features=X_train_struct.shape[1]
    )
    print(f"Training completed in {time() - start:.2f} seconds")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("performance_and_models", "vgg_images_and_features", run_id)
    os.makedirs(base_output_dir, exist_ok=True)

    print("Evaluating validation set...")
    evaluate_model(model, (X_val_img, X_val_struct), y_val, y_val_pred, name="val", base_output_dir=base_output_dir)

    # Evaluation - Test Set
    print("Preparing test data...")
    X_img_test_tensor = torch.tensor(X_test_img, dtype=torch.float32)
    if X_img_test_tensor.ndim == 4 and X_img_test_tensor.shape[1] != 3:
        X_img_test_tensor = X_img_test_tensor.permute(0, 3, 1, 2)
    X_img_test_tensor = X_img_test_tensor / 255.0

    X_struct_test_tensor = torch.tensor(X_test_struct, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_img_test_tensor, X_struct_test_tensor, y_test_tensor), batch_size=64)
    y_test_pred = []

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model.to(device)
        model.eval()
        with torch.no_grad():
            for xb_img, xb_struct, _ in test_loader:
                xb_img = xb_img.to(device, non_blocking=True)
                xb_struct = xb_struct.to(device, non_blocking=True)
                outputs = model(xb_img, xb_struct)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                y_test_pred.extend(preds)

    except RuntimeError as e:
        print(f"[WARNING] CUDA error: {e}")
        print("Switching to CPU and rebuilding model...")
        model = SmallVGG11WithFeatures(num_structured_features=X_test_struct.shape[1])
        model.load_state_dict(torch.load(os.path.join(base_output_dir, "model.pt"), map_location="cpu"))
        model.to("cpu")
        model.eval()
        with torch.no_grad():
            for xb_img, xb_struct, _ in test_loader:
                outputs = model(xb_img, xb_struct)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                y_test_pred.extend(preds)

    print("Evaluating test predictions...")
    evaluate_model(model, (X_test_img, X_test_struct), y_test, y_test_pred, name="test", base_output_dir=base_output_dir)
    print(f"All results saved in: {base_output_dir}")


if __name__ == "__main__":
    main()

