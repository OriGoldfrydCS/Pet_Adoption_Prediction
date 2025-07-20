"""
This script includes two independent CNN training modules:

1. Training a CNN on **image data only**
2. Training a CNN on **image data + structured features**

To run one of them, simply comment out the other.

Each module:
- Defines its own model architecture
- Trains the model with validation
- Applies early stopping
- Saves the best model and logs
"""


# === CNN on image data only ===

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Detect GPU or fallback to CPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'GPU detected:' if torch.cuda.is_available() else 'No GPU detected.'} {device}")

# --- Simple CNN model for 128x128 RGB images ---
class CNNModel(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # for 128x128 input
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- Train CNN using only image DataLoaders ---
def train_and_evaluate(train_loader, val_loader, epochs=5, model_save_path="best_model.pt"):
    """
    Trains CNN model using only image data from DataLoaders.
    Implements early stopping and saves best-performing model.
    """
    model = CNNModel(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    no_improve_epochs = 0
    patience = 5

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        print(f"Epoch {epoch+1} - Val Acc: {val_acc:.4f} | F1 (macro): {val_f1:.4f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to: {model_save_path}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    # --- Save loss log ---
    run_id = model_save_path.split("/")[-1].replace(".pt", "")
    log_dir = os.path.join("performance_and_models", "cnn_images", run_id)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "loss_log.txt"), "w") as f:
        for i in range(len(train_losses)):
            f.write(f"Epoch {i+1}: Train Loss = {train_losses[i]:.4f} | Val Loss = {val_losses[i]:.4f}\n")

    print(f"Saved loss log to: {log_dir}/loss_log.txt")
    return model




# === CNN with image + structured data ===

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# --- Detect device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'GPU detected:' if torch.cuda.is_available() else 'No GPU detected.'} {device}")

# --- Custom loss function to handle class imbalance ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

# --- Combined CNN model for images + structured features ---
class CNNWithFeatures(nn.Module):
    def __init__(self, num_structured_features, num_classes=4):
        super(CNNWithFeatures, self).__init__()

        # --- CNN branch ---
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.cnn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4)
        )

        # --- Structured branch ---
        self.feature_fc = nn.Sequential(
            nn.Linear(num_structured_features, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128)
        )

        # --- Fusion and classification ---
        self.combined_fc = nn.Sequential(
            nn.Linear(256 + 128, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_struct):
        x = self.cnn_fc(self.conv_layers(x_img))
        f = self.feature_fc(x_struct)
        combined = torch.cat((x, f), dim=1)
        return self.combined_fc(combined)

# --- Training function for images + features ---
def train_and_evaluate(X_img_train, X_struct_train, y_train,
                       X_img_val, X_struct_val, y_val,
                       num_structured_features,
                       epochs=30, batch_size=64, model_save_path="best_model.pt"):
    """
    Trains CNN model on combined image and structured features.
    Returns trained model and validation predictions.
    """
    def prepare_images(x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x / 255.0
        if x.ndim == 4 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        return x

    def to_tensor(x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype) if not torch.is_tensor(x) else x

    # --- Preprocessing ---
    X_img_train, X_img_val = prepare_images(X_img_train), prepare_images(X_img_val)
    X_struct_train, X_struct_val = to_tensor(X_struct_train), to_tensor(X_struct_val)
    y_train, y_val = to_tensor(y_train, dtype=torch.long), to_tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_img_train, X_struct_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_img_val, X_struct_val, y_val), batch_size=batch_size)

    model = CNNWithFeatures(num_structured_features=num_structured_features, num_classes=4).to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_val_acc = 0.0
    all_preds = []
    patience_counter = 0
    early_stopping_patience = 7

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for xb_img, xb_struct, yb in loop:
            xb_img, xb_struct, yb = xb_img.to(device), xb_struct.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb_img, xb_struct)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # --- Validation ---
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for xb_img, xb_struct, yb in val_loader:
                xb_img, xb_struct, yb = xb_img.to(device), xb_struct.to(device), yb.to(device)
                logits = model(xb_img, xb_struct)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(yb.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        print(f"Epoch {epoch+1} - Val Acc: {val_acc:.4f} | F1 (macro): {val_f1:.4f}")

        scheduler.step(epoch + 1)
        all_preds = val_preds

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved to: {model_save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    return model, all_preds
