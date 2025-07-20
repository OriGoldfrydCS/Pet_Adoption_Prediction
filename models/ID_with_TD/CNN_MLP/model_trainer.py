"""
Model and training logic for a CNN + MLP architecture that combines image data
and TF-IDF text features.

- CNN processes RGB images into a 128D vector.
- MLP processes 3000D TF-IDF text into a 128D vector.
- Features are concatenated and classified with a fully connected head.
- Includes training loop with early stopping based on validation loss.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm


# ===== Model Definition =====

class CNN_MLP(nn.Module):
    """
    CNN + MLP model for multimodal classification (image + text).

    Image path:
        - 3-layer CNN with ReLU and max-pooling.
        - Output shape: [B, 128]

    Text path:
        - 2-layer MLP with dropout and ReLU.
        - Output shape: [B, 128]

    Final classifier:
        - Concatenates both paths and outputs class logits.
    """
    def __init__(self, num_classes=4, text_dim=3000, dropout=0.3):
        super().__init__()

        # CNN branch for image input
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 64, 64]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 32, 32]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # [B, 128, 1, 1]
        )

        # MLP branch for TF-IDF text input
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Fusion and classification head
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_text):
        img_features = self.cnn(x_img).view(x_img.size(0), -1)       # [B, 128]
        text_features = self.text_branch(x_text)                     # [B, 128]
        x = torch.cat([img_features, text_features], dim=1)          # [B, 256]
        return self.classifier(x)                                    # [B, num_classes]


# ===== Training Function =====

def train_and_evaluate(X_img_train, X_text_train, y_train,
                       X_img_val, X_text_val, y_val,
                       epochs=40, batch_size=64, patience=5):
    """
    Train CNN + MLP model with early stopping on validation loss.

    Args:
        - X_img_train, X_text_train, y_train: training data
        - X_img_val, X_text_val, y_val: validation data
        - epochs (int): max epochs to train
        - batch_size (int): training batch size
        - patience (int): early stopping patience

    Returns:
        model (nn.Module): best model
        np.array: predictions on validation set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert all arrays to torch tensors
    X_img_train = torch.tensor(X_img_train, dtype=torch.float32)
    X_text_train = torch.tensor(X_text_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_img_val = torch.tensor(X_img_val, dtype=torch.float32)
    X_text_val = torch.tensor(X_text_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Create PyTorch DataLoaders
    train_loader = DataLoader(TensorDataset(X_img_train, X_text_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_img_val, X_text_val, y_val), batch_size=batch_size)

    # Initialize model, loss, optimizer
    model = CNN_MLP(num_classes=4, text_dim=X_text_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # ===== Training Loop =====
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_img, X_text, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            X_img, X_text, y_batch = X_img.to(device), X_text.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_img, X_text)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ===== Validation Loop =====
        model.eval()
        val_loss = 0.0
        all_preds = []

        with torch.no_grad():
            for X_img, X_text, y_batch in val_loader:
                X_img, X_text, y_batch = X_img.to(device), X_text.to(device), y_batch.to(device)
                logits = model(X_img, X_text)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Load best model weights
    if best_model_state:
        model.load_state_dict(best_model_state)

    # ===== Final validation prediction =====
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_img, X_text, _ in val_loader:
            X_img, X_text = X_img.to(device), X_text.to(device)
            logits = model(X_img, X_text)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(y_val.cpu().numpy(), all_preds)
    f1 = f1_score(y_val.cpu().numpy(), all_preds, average="macro")
    print(f"Validation Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    return model, np.array(all_preds)
