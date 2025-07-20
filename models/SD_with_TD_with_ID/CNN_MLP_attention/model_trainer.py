"""
CNN + MLP + Attention Model Trainer and Evaluator with TF-IDF-based text data and structured data with mean of images per sample.

Model Architecture:
-------------------
Input:
- Image tensor of shape (B, 3, H, W)
- Text feature vector (TF-IDF) of shape (B, 3000)
- Structured feature vector of shape (B, 15)

Branches:
1. Image Branch (CNN):
   - Conv2D → ReLU → MaxPool
   - Conv2D → ReLU → MaxPool
   - Conv2D → ReLU → AdaptiveAvgPool2D
   → Output: (B, 128)

2. Text Branch (MLP):
   - Linear(3000 → 256) → ReLU → Dropout
   - Linear(256 → 128) → ReLU
   → Output: (B, 128)

3. Structured Branch (MLP):
   - Linear(15 → 64) → ReLU
   - Linear(64 → 128) → ReLU
   → Output: (B, 128)

4. Attention Layer:
   - MultiheadAttention with 4 heads
   - Query: structured vector (1 token)
   - Key/Value: text vector (1 token)
   → Output: (B, 128)

Fusion:
- Concatenate: [image vector, attended text]
- Classifier MLP:
    Linear(256 → 128) → ReLU → Dropout → Linear(128 → num_classes)

The training function includes:
- Early stopping based on validation loss
- Batch-wise training and evaluation
- Accuracy and F1 (macro) reporting
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm


class CNN_MLP_Attention(nn.Module):
    def __init__(self, num_classes=4, text_dim=3000, struct_dim=15, dropout=0.3):
        super().__init__()

        # === Image branch ===
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Output shape: (B, 128, 1, 1)
        )

        # === Text branch ===
        self.text_proj1 = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.text_proj2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # === Structured branch ===
        self.struct_branch = nn.Sequential(
            nn.Linear(struct_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # === Attention: structured queries over text ===
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        # === Final classifier ===
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_text, x_struct):
        # Image branch
        img_features = self.cnn(x_img).view(x_img.size(0), -1)  # (B, 128)

        # Text branch
        x_text_1 = self.text_proj1(x_text)  # (B, 256)
        x_text_2 = self.text_proj2(x_text_1)  # (B, 128)

        # Structured branch
        struct_feat = self.struct_branch(x_struct)  # (B, 128)

        # Add sequence dimension for attention
        x_text_2 = x_text_2.unsqueeze(1)       # (B, 1, 128)
        struct_feat = struct_feat.unsqueeze(1)  # (B, 1, 128)

        # Attention mechanism
        attended, _ = self.attn(query=struct_feat, key=x_text_2, value=x_text_2)  # (B, 1, 128)
        attended = attended.squeeze(1)  # (B, 128)

        # Combine image and attended text
        x = torch.cat([img_features, attended], dim=1)  # (B, 256)
        return self.classifier(x)


def train_and_evaluate(X_img_train, X_text_train, X_struct_train, y_train,
                       X_img_val, X_text_val, X_struct_val, y_val,
                       epochs=50, batch_size=64, patience=5):
    """
    Trains the CNN_MLP_Attention model and evaluates on validation set.
    Returns the best model and predictions on the validation set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Convert data to tensors ===
    X_img_train = torch.tensor(X_img_train, dtype=torch.float32)
    X_text_train = torch.tensor(X_text_train, dtype=torch.float32)
    X_struct_train = torch.tensor(X_struct_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_img_val = torch.tensor(X_img_val, dtype=torch.float32)
    X_text_val = torch.tensor(X_text_val, dtype=torch.float32)
    X_struct_val = torch.tensor(X_struct_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # === Dataloaders ===
    train_loader = DataLoader(TensorDataset(X_img_train, X_text_train, X_struct_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_img_val, X_text_val, X_struct_val, y_val),
                            batch_size=batch_size)

    # === Model initialization ===
    model = CNN_MLP_Attention(
        num_classes=4,
        text_dim=X_text_train.shape[1],
        struct_dim=X_struct_train.shape[1]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # === Training loop with early stopping ===
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_img, X_text, X_struct, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            X_img, X_text, X_struct, y_batch = X_img.to(device), X_text.to(device), X_struct.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_img, X_text, X_struct)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # === Validation ===
        model.eval()
        val_loss = 0.0
        all_preds = []
        with torch.no_grad():
            for X_img, X_text, X_struct, y_batch in val_loader:
                X_img, X_text, X_struct, y_batch = X_img.to(device), X_text.to(device), X_struct.to(device), y_batch.to(device)
                logits = model(X_img, X_text, X_struct)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        # === Early stopping logic ===
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # === Load best model ===
    if best_model_state:
        model.load_state_dict(best_model_state)

    # === Final predictions on validation set ===
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_img, X_text, X_struct, _ in val_loader:
            X_img, X_text, X_struct = X_img.to(device), X_text.to(device), X_struct.to(device)
            logits = model(X_img, X_text, X_struct)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(y_val.cpu().numpy(), all_preds)
    f1 = f1_score(y_val.cpu().numpy(), all_preds, average="macro")
    print(f"Validation Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    return model, np.array(all_preds)
