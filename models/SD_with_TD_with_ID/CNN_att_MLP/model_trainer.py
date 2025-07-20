"""
Multimodal classification model: CNN + Attention + MLP

This script defines and trains a deep learning model that combines:
- Image sequences (multiple images per sample)
- TF-IDF text vectors
- Structured data

Architecture:
- CNN extracts features per image
- Attention aggregates features across image sequence
- Text and structured branches processed separately
- Final MLP classifier over fused representation

Includes:
- Padding & masking of image sequences
- Training with early stopping on validation loss
- Evaluation with accuracy and F1 metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm


# ====== Multimodal Model Definition ======
class CNN_Att_MLP(nn.Module):
    """
    CNN + Attention + MLP classifier.
    Processes:
    - Image sequences via CNN + self-attention
    - TF-IDF text via MLP
    - Structured features via MLP
    Combines all into a final classifier.
    """
    def __init__(self, text_dim=3000, struct_dim=15, img_feature_dim=128, num_classes=4, dropout=0.3):
        super().__init__()

        # CNN for each image
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # → [128, 1, 1]
        )

        # Text branch
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Structured data branch
        self.struct_branch = nn.Sequential(
            nn.Linear(struct_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Attention over image sequence
        self.image_att = nn.MultiheadAttention(embed_dim=img_feature_dim, num_heads=4, batch_first=True)

        # Final fusion MLP
        self.classifier = nn.Sequential(
            nn.Linear(img_feature_dim + 128 + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_text, x_struct, image_mask=None):
        B, N, H, W, C = x_img.shape  # [B, N, 128, 128, 3]
        x_img = x_img.permute(0, 1, 4, 2, 3)         # [B, N, 3, 128, 128]
        x_img = x_img.view(B * N, 3, H, W)           # [B*N, 3, 128, 128]
        img_feats = self.cnn(x_img).squeeze(-1).squeeze(-1)  # [B*N, 128]
        img_feats = img_feats.view(B, N, -1)         # [B, N, 128]

        # Attention across image sequence
        att_out, _ = self.image_att(img_feats, img_feats, img_feats, key_padding_mask=image_mask)
        img_repr = att_out.mean(dim=1)               # [B, 128]

        # Process other modalities
        text_repr = self.text_branch(x_text)         # [B, 128]
        struct_repr = self.struct_branch(x_struct)   # [B, 32]

        # Concatenate and classify
        x = torch.cat([img_repr, text_repr, struct_repr], dim=1)
        return self.classifier(x)


# ====== Padding Image Sequences ======
def pad_image_sequences(image_lists, max_len=5):
    """
    Pads each image sequence to `max_len` and creates a mask.
    """
    batch_size = len(image_lists)
    C, H, W = image_lists[0][0].shape

    padded = torch.zeros((batch_size, max_len, C, H, W), dtype=torch.float32)
    mask = torch.ones((batch_size, max_len), dtype=torch.bool)

    for i, seq in enumerate(image_lists):
        truncated = seq[:max_len]

        # Convert numpy → float tensor if needed
        truncated_tensors = [
            torch.from_numpy(img).float() if isinstance(img, np.ndarray) else img.float()
            for img in truncated
        ]

        padded[i, :len(truncated_tensors)] = torch.stack(truncated_tensors)
        mask[i, :len(truncated_tensors)] = False  # False means real image

    return padded, mask


# ====== Training Pipeline ======
def train_and_evaluate(X_img_train, X_text_train, X_struct_train, y_train,
                       X_img_val, X_text_val, X_struct_val, y_val,
                       epochs=20, batch_size=32, patience=5):
    """
    Trains the CNN+Attn+MLP model with early stopping.
    Returns:
        model: trained model (best on val)
        y_val_pred: predictions on validation set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Pad image sequences and get attention masks
    X_img_train, mask_train = pad_image_sequences(X_img_train)
    X_img_val, mask_val = pad_image_sequences(X_img_val)

    # Convert to tensors
    X_text_train = torch.tensor(X_text_train, dtype=torch.float32)
    X_struct_train = torch.tensor(X_struct_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_text_val = torch.tensor(X_text_val, dtype=torch.float32)
    X_struct_val = torch.tensor(X_struct_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Dataloaders
    train_loader = DataLoader(TensorDataset(X_img_train, X_text_train, X_struct_train, mask_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_img_val, X_text_val, X_struct_val, mask_val, y_val),
                            batch_size=batch_size)

    # Init model
    model = CNN_Att_MLP(
        num_classes=4,
        text_dim=X_text_train.shape[1],
        struct_dim=X_struct_train.shape[1]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_img, X_text, X_struct, mask, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            X_img, X_text, X_struct, mask, y_batch = X_img.to(device), X_text.to(device), X_struct.to(device), mask.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_img, X_text, X_struct, image_mask=mask)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        with torch.no_grad():
            for X_img, X_text, X_struct, mask, y_batch in val_loader:
                X_img, X_text, X_struct, mask, y_batch = X_img.to(device), X_text.to(device), X_struct.to(device), mask.to(device), y_batch.to(device)
                logits = model(X_img, X_text, X_struct, image_mask=mask)
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

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Final validation predictions
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_img, X_text, X_struct, mask, _ in val_loader:
            X_img, X_text, X_struct, mask = X_img.to(device), X_text.to(device), X_struct.to(device), mask.to(device)
            logits = model(X_img, X_text, X_struct, image_mask=mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(y_val.cpu().numpy(), all_preds)
    f1 = f1_score(y_val.cpu().numpy(), all_preds, average="macro")
    print(f"Validation Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    return model, np.array(all_preds)
