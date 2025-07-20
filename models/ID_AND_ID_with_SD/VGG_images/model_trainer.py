"""
This script includes two independent CNN training modules:

1. Training a VGG on **image data only**
2. Training a VGG on **image data + structured features**

To run one of them, simply comment out the other.

Each module:
- Defines its own model architecture
- Trains the model with validation
- Applies early stopping
- Saves the best model and logs
"""

# === VGG19 on image data only ===

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'GPU detected:' if torch.cuda.is_available() else 'No GPU detected.'} {device}")

# VGG19 architecture with custom classifier head, last convolutional block is unfrozen for fine-tuning.
def build_model(num_classes=4):
    vgg19 = models.vgg19(pretrained=True)

   # Unfreeze last conv block
    for name, param in vgg19.features.named_parameters():
        if "36" in name or "35" in name or "34" in name:
            param.requires_grad = True


    # classifier
    vgg19.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(512, num_classes)
    )


    return vgg19

# Train and evaluate function
def train_and_evaluate(train_loader, val_loader, epochs=3, model_save_path="best_model_vgg19.pt"):
    """
    Trains a VGG19 model on image data only.

    Args:
        train_loader: DataLoader for training images and labels.
        val_loader: DataLoader for validation images and labels.
        epochs: Number of training epochs.
        model_save_path: Path to save the best model weights.

    Returns:
        Trained PyTorch model with best validation accuracy.
    """
    model = build_model(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_val_acc = 0.0
    epochs_no_improve = 0
    early_stop_patience = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                outputs = model(xb)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | F1 (macro): {val_f1:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to: {model_save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
                break

    return model





# === VGG11 with image + structured data ===

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, log_loss
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'GPU detected:' if torch.cuda.is_available() else 'No GPU detected.'} {device}")

# Hybrid model:
# - VGG11 (frozen conv layers) for image feature extraction
# - MLP for structured features
# - Combined in a shared dense layer before classification
class SmallVGG11WithFeatures(nn.Module):
    def __init__(self, num_structured_features, num_classes=4):
        super(SmallVGG11WithFeatures, self).__init__()

        self.vgg11 = models.vgg11(pretrained=True).features
        for param in self.vgg11.parameters():
            param.requires_grad = False

        # Dynamically determine conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            out = self.vgg11(dummy)
            self.flattened_size = out.view(1, -1).size(1)

        # Smaller FC layers
        self.cnn_fc = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.feature_fc = nn.Sequential(
            nn.Linear(num_structured_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_struct):
        x = self.vgg11(x_img)
        x = torch.flatten(x, 1)
        x = self.cnn_fc(x)

        f = self.feature_fc(x_struct)
        combined = torch.cat((x, f), dim=1)
        return self.combined_fc(combined)


# Training function
def train_and_evaluate(X_img_train, X_struct_train, y_train,
                       X_img_val, X_struct_val, y_val,
                       num_structured_features,
                       epochs=10, batch_size=32, model_save_path="best_model_vgg11_small.pt"):
    """
    Trains a hybrid VGG11 + MLP model using both images and structured features.

    Args:
        X_img_train, X_img_val: Image data (numpy or tensors).
        X_struct_train, X_struct_val: Structured numeric features.
        y_train, y_val: True labels.
        num_structured_features: Number of structured features (input size to MLP).
        epochs, batch_size: Training settings.
        model_save_path: Where to save the best model weights.

    Returns:
        Trained model and best predictions on validation set.
    """

    def to_tensor(x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype) if not torch.is_tensor(x) else x

    def safe_permute(x):
        return x.permute(0, 3, 1, 2) if x.shape[1] != 3 else x

    # Preprocess inputs
    X_img_train = safe_permute(to_tensor(X_img_train))
    X_img_val = safe_permute(to_tensor(X_img_val))
    X_struct_train = to_tensor(X_struct_train)
    X_struct_val = to_tensor(X_struct_val)
    y_train = to_tensor(y_train, dtype=torch.long)
    y_val = to_tensor(y_val, dtype=torch.long)

    # Dataloaders
    train_loader = DataLoader(TensorDataset(X_img_train, X_struct_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_img_val, X_struct_val, y_val), batch_size=batch_size)

    # Model
    model = SmallVGG11WithFeatures(num_structured_features=num_structured_features).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    best_val_acc = 0.0
    best_preds = []

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for xb_img, xb_struct, yb in loop:
            xb_img, xb_struct, yb = xb_img.to(device), xb_struct.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb_img, xb_struct)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for xb_img, xb_struct, yb in val_loader:
                xb_img, xb_struct, yb = xb_img.to(device), xb_struct.to(device), yb.to(device)
                outputs = model(xb_img, xb_struct)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_preds.extend(preds)
                all_labels.extend(yb.cpu().numpy())
                all_probs.append(probs)

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        val_logloss = log_loss(all_labels, np.vstack(all_probs))

        print(f"Epoch {epoch+1} - Val Acc: {val_acc:.4f} | F1 (macro): {val_f1:.4f} | LogLoss: {val_logloss:.4f}")
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_preds = all_preds
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to: {model_save_path}")

    print(f"Training finished. Best val accuracy: {best_val_acc:.4f}")
    return model, best_preds