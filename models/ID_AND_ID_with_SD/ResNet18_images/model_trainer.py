"""
This script includes two independent CNN training modules:

1. Training a ResNet18 on **image data only**
2. Training a ResNet18 on **image data + structured features**

To run one of them, simply comment out the other.

Each module:
- Defines its own model architecture
- Trains the model with validation
- Applies early stopping
- Saves the best model and logs
"""

# === ResNet18 on image data only ===

import os
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'GPU detected:' if torch.cuda.is_available() else 'No GPU detected.'} {device}")

def build_model(num_classes=4):
    """
    Loads pretrained ResNet18 and replaces its FC layer with:
    Linear → BatchNorm → ReLU → Dropout → Linear (output layer).
    Fine-tunes layer4 and the classifier head only.
    """
    resnet = models.resnet18(pretrained=True)

    for param in resnet.parameters():
        param.requires_grad = False

    for name, param in resnet.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True

    resnet.fc = nn.Sequential(
        nn.Linear(resnet.fc.in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    return resnet

def train_and_evaluate(train_loader, val_loader, epochs=30, model_save_path="best_model_resnet18.pt", early_stop_patience=5):
    """
    Trains the ResNet18 model and evaluates on validation set.
    Saves best model by accuracy with early stopping.
    """
    model = build_model(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_val_acc = 0.0
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

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

        print(f"Epoch {epoch+1} - Val Acc: {val_acc:.4f} | F1 (macro): {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to: {model_save_path}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s).")

        if no_improve_epochs >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return model






# === ResNet18 with image + structured data ===

"""
Model Architecture:
- ResNet18 backbone for image feature extraction (output: 512-dim vector).
- MLP for structured features (2-layer FC, output: 128-dim vector).
- Final classifier: concatenated [512 + 128] → 256 → num_classes.
- Only 'layer4' and classifier head of ResNet18 are fine-tuned; the rest is frozen.
"""


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'GPU detected:' if torch.cuda.is_available() else 'No GPU detected.'} {device}")

class ResNet18WithFeatures(nn.Module):
    """
    Combined ResNet18 + MLP model:
    - Extracts 512-dim features from images
    - Projects structured data to 128-dim
    - Concatenates and passes through 2 FC layers
    """
    def __init__(self, num_structured_features, num_classes=4):
        super().__init__()
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

        for param in base_model.parameters():
            param.requires_grad = False
        for name, param in base_model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True

        self.resnet_backbone = nn.Sequential(*list(base_model.children())[:-1])  # [B, 512, 1, 1]

        self.feature_fc = nn.Sequential(
            nn.Linear(num_structured_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_struct):
        x = self.resnet_backbone(x_img)      # [B, 512, 1, 1]
        x = torch.flatten(x, 1)              # [B, 512]
        f = self.feature_fc(x_struct)        # [B, 128]
        combined = torch.cat((x, f), dim=1)  # [B, 640]
        return self.combined_fc(combined)    # [B, num_classes]

def train_and_evaluate(X_img_train, X_struct_train, y_train,
                       X_img_val, X_struct_val, y_val,
                       num_structured_features,
                       epochs=30, batch_size=64, model_save_path="best_model_resnet18.pt"):
    """
    Trains the combined model and evaluates on val set.
    Saves best model based on accuracy.
    """
    def to_tensor(x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype) if not torch.is_tensor(x) else x

    def safe_permute(x):
        return x.permute(0, 3, 1, 2) if x.shape[1] != 3 else x

    # Convert to tensors
    X_img_train = safe_permute(to_tensor(X_img_train))
    X_img_val = safe_permute(to_tensor(X_img_val))
    X_struct_train = to_tensor(X_struct_train)
    X_struct_val = to_tensor(X_struct_val)
    y_train = to_tensor(y_train, dtype=torch.long)
    y_val = to_tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_img_train, X_struct_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_img_val, X_struct_val, y_val), batch_size=batch_size)

    model = ResNet18WithFeatures(num_structured_features=num_structured_features).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_val_acc = 0.0
    logs = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for xb_img, xb_struct, yb in loop:
            xb_img, xb_struct, yb = xb_img.to(device), xb_struct.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb_img, xb_struct)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        all_preds, all_labels = [], []
        total_val_loss = 0.0

        with torch.no_grad():
            for xb_img, xb_struct, yb in val_loader:
                xb_img, xb_struct, yb = xb_img.to(device), xb_struct.to(device), yb.to(device)
                outputs = model(xb_img, xb_struct)
                loss = criterion(outputs, yb)
                total_val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | F1 (macro): {val_f1:.4f}")

        logs["epoch"].append(epoch + 1)
        logs["train_loss"].append(avg_train_loss)
        logs["val_loss"].append(avg_val_loss)
        logs["val_acc"].append(val_acc)
        logs["val_f1"].append(val_f1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to: {model_save_path}")

    return model, all_preds, logs
