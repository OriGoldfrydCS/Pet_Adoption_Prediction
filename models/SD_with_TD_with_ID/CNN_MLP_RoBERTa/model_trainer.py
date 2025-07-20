"""
Model trainer script: Multimodal Model (Images (with 5 images) + RoBERTa + Structured)

Architecture Overview:
- Each input modality has its own dedicated branch:
    - Images are processed by a shared CNN, followed by attention over multiple images.
    - Structured features go through a small MLP and are projected to the same dimensionality as the text embeddings.
    - Text is encoded using a pre-trained model (RoBERTa) and then attended to using the structured projection as a query.

- Two attention mechanisms are used:
    - One to focus across multiple images for each sample.
    - Another to align structured information with relevant parts of the text.

Training uses standard cross-entropy loss, early stopping, and evaluation on accuracy and macro-F1.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm


# Model Definition
class CNN_MLP_Attention(nn.Module):
    def __init__(self, num_classes=4, text_embed_dim=768, struct_dim=15, dropout=0.3):
        super().__init__()

        # Image processing CNN (shared across all images)
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

        # Structured features → 2-layer MLP
        self.struct_branch = nn.Sequential(
            nn.Linear(struct_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.struct_proj = nn.Linear(128, text_embed_dim)  # Project to same dim as text embeddings

        # Attention: structured features attend over text tokens
        self.attn = nn.MultiheadAttention(embed_dim=text_embed_dim, num_heads=4, batch_first=True)

        # Attention over multiple images (same modality)
        self.img_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(128 + text_embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_imgs, x_text, x_struct):
        # Input shape expected: (B, N, 3, 128, 128)
        if x_imgs.dim() != 5:
            raise ValueError(f"Expected input shape (B, N, 3, 128, 128), but got {x_imgs.shape}")
        if x_imgs.shape[2] == 128 and x_imgs.shape[3] == 3:
            x_imgs = x_imgs.permute(0, 1, 3, 2, 4)  # Fix shape: (B, N, H, C, W) → (B, N, C, H, W)

        B, N, C, H, W = x_imgs.shape
        x_imgs = x_imgs.view(B * N, C, H, W)  # Flatten image batch for CNN

        # CNN encoding → reshape → attention over images
        img_feats = self.cnn(x_imgs)                  # (B*N, 128, 1, 1)
        img_feats = img_feats.flatten(start_dim=1)    # (B*N, 128)
        img_feats = img_feats.view(B, N, -1)          # (B, N, 128)

        query = img_feats.mean(dim=1).unsqueeze(1)    # Global image context (B, 1, 128)
        attn_out, _ = self.img_attn(query=query, key=img_feats, value=img_feats)
        img_features = attn_out.squeeze(1)            # (B, 128)

        # Structured → MLP → project to match text dimension
        struct_feat = self.struct_branch(x_struct)         # (B, 128)
        struct_feat = self.struct_proj(struct_feat).unsqueeze(1)  # (B, 1, 768)

        # Attention: structured features query over token embeddings
        attended_text, _ = self.attn(query=struct_feat, key=x_text, value=x_text)
        attended_text = attended_text.squeeze(1)  # (B, 768)

        # Combine image + attended text
        x = torch.cat([img_features, attended_text], dim=1)
        return self.classifier(x)

# Training Loop
def train_and_evaluate(X_img_train, X_text_train, X_struct_train, y_train,
                       X_img_val, X_text_val, X_struct_val, y_val,
                       epochs=30, batch_size=64, patience=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert numpy arrays to tensors and fix image shape
    X_img_train = torch.tensor(X_img_train, dtype=torch.float32).permute(0, 1, 4, 2, 3)
    X_text_train = torch.tensor(X_text_train, dtype=torch.float32)
    X_struct_train = torch.tensor(X_struct_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_img_val = torch.tensor(X_img_val, dtype=torch.float32).permute(0, 1, 4, 2, 3)
    X_text_val = torch.tensor(X_text_val, dtype=torch.float32)
    X_struct_val = torch.tensor(X_struct_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Dataloaders for batching
    train_loader = DataLoader(TensorDataset(X_img_train, X_text_train, X_struct_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_img_val, X_text_val, X_struct_val, y_val),
                            batch_size=batch_size)

    # Initialize model
    model = CNN_MLP_Attention(
        num_classes=4,
        text_embed_dim=X_text_train.shape[2],
        struct_dim=X_struct_train.shape[1]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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

        # Validation phase
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

    if best_model_state:
        model.load_state_dict(best_model_state)

    # Final predictions
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
