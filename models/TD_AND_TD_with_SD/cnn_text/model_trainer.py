"""
CNN model for BERT token embeddings combined with structured data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm


class CNNWithStructured(nn.Module):
    """
    A 1D CNN model that processes token embeddings and structured data jointly.

    Args:
        - input_channels (int): Number of channels in token embeddings (default: 768)
        - seq_len (int): Length of token sequence (default: 128)
        - structured_dim (int): Dimension of structured input
        - num_classes (int): Number of output classes
    """
    def __init__(self, input_channels=768, seq_len=128, structured_dim=14, num_classes=4):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 + structured_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_tokens, x_structured):
        """
        Forward pass.

        Args:
            - x_tokens (Tensor): Token input [B, 768, 128]
            - x_structured (Tensor): Structured input [B, structured_dim]

        Returns:
            - Tensor: Class logits [B, num_classes]
        """
        x = self.cnn(x_tokens).squeeze(2)         # [B, 128]
        x = torch.cat([x, x_structured], dim=1)   # [B, 128 + structured_dim]
        return self.fc(x)


def train_and_evaluate(X_tokens_train, X_struct_train, y_train,
                       X_tokens_val, X_struct_val, y_val,
                       epochs=40, batch_size=64, patience=5):
    """
    Trains the CNNWithStructured model and evaluates on the validation set.

    Args:
        - X_tokens_train (np.ndarray): Token input for training
        - X_struct_train (np.ndarray): Structured input for training
        - y_train (np.ndarray): Labels for training
        - X_tokens_val (np.ndarray): Token input for validation
        - X_struct_val (np.ndarray): Structured input for validation
        - y_val (np.ndarray): Labels for validation
        - epochs (int): Number of training epochs
        - batch_size (int): Batch size
        - patience (int): Early stopping patience

    Returns:
        - model (nn.Module): Trained model
        - y_val_pred (np.ndarray): Predicted labels on validation set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert inputs to tensors
    X_tokens_train = torch.tensor(X_tokens_train, dtype=torch.float32)
    X_struct_train = torch.tensor(X_struct_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_tokens_val = torch.tensor(X_tokens_val, dtype=torch.float32)
    X_struct_val = torch.tensor(X_struct_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Build datasets
    train_dataset = TensorDataset(X_tokens_train, X_struct_train, y_train)
    val_dataset = TensorDataset(X_tokens_val, X_struct_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = CNNWithStructured(
        input_channels=768,
        seq_len=128,
        structured_dim=X_struct_train.shape[1]
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

        for X_tok, X_struct, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            X_tok, X_struct, y_batch = X_tok.to(device), X_struct.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_tok, X_struct)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_tok, X_struct, y_batch in val_loader:
                X_tok, X_struct, y_batch = X_tok.to(device), X_struct.to(device), y_batch.to(device)
                logits = model(X_tok, X_struct)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()

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

    # Final prediction on validation set
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_tok, X_struct, _ in val_loader:
            X_tok, X_struct = X_tok.to(device), X_struct.to(device)
            logits = model(X_tok, X_struct)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(y_val.cpu().numpy(), all_preds)
    f1 = f1_score(y_val.cpu().numpy(), all_preds, average="macro")
    print(f"Validation Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    return model, np.array(all_preds)
