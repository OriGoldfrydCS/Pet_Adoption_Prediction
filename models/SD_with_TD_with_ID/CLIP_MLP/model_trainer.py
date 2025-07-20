"""
This script defines and trains a multimodal classification model based on:
- CLIP image embeddings (sequence of 10 vectors per sample/or 1 if it is averaged)
- CLIP text embeddings (single 512-dim vector)
- Structured numeric features

Architecture:
- Uses multi-head self-attention over image embeddings
- Concatenates attended image vector with text and structured features
- Passes the combined vector through an MLP for classification

Includes:
- Early stopping
- Accuracy and F1 evaluation on validation set
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm


# ====== CLIP + MLP Model with Image Attention ======
class CLIP_MLP_Attn(nn.Module):
    """
    Multimodal classifier using:
    - Attention over CLIP image embeddings
    - CLIP text embedding
    - Structured features
    """
    def __init__(self, struct_dim, dropout=0.1, hidden_dim=512, num_classes=4):
        super().__init__()

        # Attention over image sequence [B, 10, 512]
        self.image_attn = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)

        # Final classifier over [image_attn + text + structured]
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512 + struct_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_img, x_text, x_struct):
        # Apply attention to image embeddings
        query = x_img.mean(dim=1).unsqueeze(1)  # [B, 1, 512]
        attn_out, _ = self.image_attn(query=query, key=x_img, value=x_img)  # [B, 1, 512]
        img_repr = attn_out.squeeze(1)  # [B, 512]

        # Concatenate image, text, structured features
        x = torch.cat([img_repr, x_text, x_struct], dim=1)
        return self.classifier(x)


# ====== Training and Evaluation Loop ======
def train_and_evaluate(X_img_train, X_text_train, X_struct_train, y_train,
                       X_img_val, X_text_val, X_struct_val, y_val,
                       epochs=100, batch_size=64, patience=5):
    """
    Trains the CLIP+MLP+Attention model using a validation split.
    Early stopping is based on validation loss.
    Returns:
        - Trained model
        - Final validation predictions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert numpy arrays to torch tensors
    X_img_train = torch.tensor(X_img_train, dtype=torch.float32)
    X_text_train = torch.tensor(X_text_train, dtype=torch.float32)
    X_struct_train = torch.tensor(X_struct_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_img_val = torch.tensor(X_img_val, dtype=torch.float32)
    X_text_val = torch.tensor(X_text_val, dtype=torch.float32)
    X_struct_val = torch.tensor(X_struct_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_img_train, X_text_train, X_struct_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_img_val, X_text_val, X_struct_val, y_val),
                            batch_size=batch_size)

    # Initialize model
    model = CLIP_MLP_Attn(struct_dim=X_struct_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # Training loop with early stopping
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_img, X_text, X_struct, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            X_img, X_text, X_struct, y_batch = X_img.to(device), X_text.to(device), X_struct.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_img, X_text, X_struct)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation loop
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

        # Check early stopping condition
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Final validation predictions
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
