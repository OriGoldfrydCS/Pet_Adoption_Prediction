"""
Train and evaluate an MLP classifier on flattened BERT token embeddings
(optionally combined with structured data).

Includes:
- MLP architecture with batch norm and dropout
- Training loop with early stopping
- Final evaluation on validation set
- Deterministic behavior with fixed random seed
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import random
from tqdm import tqdm

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Define the MLP model
class BertTokenMLP(nn.Module):
    """
    Multi-layer perceptron for classification based on BERT token embeddings
    (optionally concatenated with structured data).
    """
    def __init__(self, input_dim, num_classes=4):
        super(BertTokenMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def train_and_evaluate(X_train, y_train, X_val, y_val, epochs=20, batch_size=64, patience=5):
    """
    Train the MLP model on training data and evaluate on validation set.

    Args:
        - X_train (np.array): Training input features.
        - y_train (np.array): Training labels.
        - X_val (np.array): Validation input features.
        - y_val (np.array): Validation labels.
        - epochs (int): Maximum number of training epochs.
        - batch_size (int): Batch size for training.
        - patience (int): Patience for early stopping.

    Returns:
        - model (BertTokenMLP): Trained model.
        - np.array: Predictions on validation set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val_tensor), batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss
    input_dim = X_train.shape[1]
    model = BertTokenMLP(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop with early stopping
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Early stopping check
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
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on validation set
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(y_val, all_preds)
    f1 = f1_score(y_val, all_preds, average="macro")
    print(f"Final Validation Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    return model, np.array(all_preds)
