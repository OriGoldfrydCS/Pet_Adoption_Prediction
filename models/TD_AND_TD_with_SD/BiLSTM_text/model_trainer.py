"""
BiLSTM model for text data combined with structured data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm


class BiLSTMWithStructured(nn.Module):
    """
    A BiLSTM model that takes both BERT token embeddings (sequence input)
    and structured data, and outputs class logits.

    Args:
        input_dim (int): Dimensionality of token embeddings (default: 768)
        hidden_dim (int): LSTM hidden state size
        num_layers (int): Number of LSTM layers
        structured_dim (int): Number of structured (non-textual) data
        num_classes (int): Number of output classes
        dropout (float): Dropout probability
    """
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, structured_dim=1550 - (768 * 2),
                 num_classes=4, dropout=0.3):
        super(BiLSTMWithStructured, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + structured_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_seq, x_structured):
        """
        Forward pass through the BiLSTM model.
        Args:
            - x_seq (Tensor): Token sequence input [batch, seq_len, 768]
            - x_structured (Tensor): Structured input [batch, structured_dim]

        Returns:
            - logits (Tensor): Class scores [batch, num_classes]
        """
        output, _ = self.lstm(x_seq)                 # LSTM output: [batch, seq_len, 2*hidden]
        output = output[:, -1, :]                    # Use last timestep: [batch, 2*hidden]
        x = torch.cat([output, x_structured], dim=1) # Concatenate with structured features
        x = self.dropout(x)
        return self.fc(x)                            # Output logits


def train_and_evaluate(X_seq_train, X_struct_train, y_train,
                       X_seq_val, X_struct_val, y_val,
                       epochs=40, batch_size=64, patience=5):
    """
    Trains the BiLSTMWithStructured model and evaluates on the validation set.

    Args:
        - X_seq_train (array): Training token sequences
        - X_struct_train (array): Training structured data
        - y_train (array): Training labels
        - X_seq_val (array): Validation token sequences
        - X_struct_val (array): Validation structured data
        - y_val (array): Validation labels
        - epochs (int): Max training epochs
        - batch_size (int): Batch size
        - patience (int): Early stopping patience

    Returns:
        - model (nn.Module): Trained model
        - y_val_pred (np.array): Predicted labels for validation set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert to tensors
    X_seq_train = torch.tensor(X_seq_train, dtype=torch.float32)
    X_struct_train = torch.tensor(X_struct_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_seq_val = torch.tensor(X_seq_val, dtype=torch.float32)
    X_struct_val = torch.tensor(X_struct_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_seq_train, X_struct_train, y_train)
    val_dataset = TensorDataset(X_seq_val, X_struct_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    input_dim = X_seq_train.shape[2]
    structured_dim = X_struct_train.shape[1]
    model = BiLSTMWithStructured(input_dim=input_dim, structured_dim=structured_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_seq, X_struct, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            X_seq, X_struct, y_batch = X_seq.to(device), X_struct.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_seq, X_struct)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_seq, X_struct, y_batch in val_loader:
                X_seq, X_struct, y_batch = X_seq.to(device), X_struct.to(device), y_batch.to(device)
                logits = model(X_seq, X_struct)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

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

    # Load best weights
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Final validation predictions
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_seq, X_struct, _ in val_loader:
            X_seq, X_struct = X_seq.to(device), X_struct.to(device)
            logits = model(X_seq, X_struct)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(y_val.cpu().numpy(), all_preds)
    f1 = f1_score(y_val.cpu().numpy(), all_preds, average="macro")
    print(f"Validation Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    return model, np.array(all_preds)
