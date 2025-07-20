"""
CNN + LSTM model for classification using image data and BERT embeddings.

- CNN processes image input into a 64D vector.
- Text input can be either:
    - BERT CLS token ([B, 768]) → used as-is
    - BERT full sequence ([B, T, 768]) → passed through an LSTM
- Final classifier receives concatenated [image + text] representation.

Includes:
- Model definition
- Training loop with early stopping based on validation loss
- Returns model, accuracy, F1 score, and validation predictions
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm

# ===== Model Definition =====

class CNN_LSTM(nn.Module):
    """
    CNN + LSTM hybrid model.

    - CNN processes 128x128 RGB images into a 64D vector.
    - BERT embeddings are passed either:
        - as-is (if shape is [B, 768], i.e., CLS token), or
        - through an LSTM (if shape is [B, T, 768]).
    - Outputs are concatenated and passed to a fully connected classifier.
    """
    def __init__(self, bert_hidden_dim=768, lstm_hidden_dim=128, num_classes=4):
        super().__init__()

        # CNN for image features
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Output: [B, 64, 1, 1]
        )

        # LSTM for BERT sequence input
        self.lstm = nn.LSTM(
            input_size=bert_hidden_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
            bidirectional=False
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 + lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_text):
        img_features = self.cnn(x_img).view(x_img.size(0), -1)  # [B, 64]

        if x_text.dim() == 2:
            # Text input is CLS token: [B, 768]
            text_features = x_text
        else:
            # Text input is full sequence: [B, T, 768]
            _, (hn, _) = self.lstm(x_text)  # hn: [1, B, hidden]
            text_features = hn.squeeze(0)  # [B, hidden]

        combined = torch.cat([img_features, text_features], dim=1)
        return self.classifier(combined)


# ===== Training Function =====

def train_cnn_lstm(X_img_train, X_text_train, y_train,
                   X_img_val, X_text_val, y_val,
                   num_classes=4, epochs=200, batch_size=64, patience=5):
    """
    Train CNN + LSTM model with early stopping on validation loss.

    Args:
        - X_img_train, X_text_train, y_train: training data
        - X_img_val, X_text_val, y_val: validation data
        - num_classes (int): number of output classes
        - epochs (int): max training epochs
        - batch_size (int): batch size
        - patience (int): early stopping patience

    Returns:
        - model (nn.Module): trained model
        - acc (float): validation accuracy
        - f1 (float): validation macro F1
        - np.ndarray: predicted validation labels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert inputs to tensors
    X_img_train = torch.tensor(X_img_train, dtype=torch.float32)
    X_text_train = torch.tensor(X_text_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_img_val = torch.tensor(X_img_val, dtype=torch.float32)
    X_text_val = torch.tensor(X_text_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_img_train, X_text_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_img_val, X_text_val, y_val_tensor), batch_size=batch_size)

    # Initialize model
    model = CNN_LSTM(bert_hidden_dim=X_text_train.shape[-1], num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_model = None
    best_loss = float('inf')
    patience_counter = 0

    # ===== Training loop =====
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_img, X_text, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            X_img, X_text, y = X_img.to(device), X_text.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X_img, X_text)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # ===== Validation =====
        model.eval()
        val_loss = 0.0
        all_preds = []
        with torch.no_grad():
            for X_img, X_text, y in val_loader:
                X_img, X_text, y = X_img.to(device), X_text.to(device), y.to(device)
                logits = model(X_img, X_text)
                loss = loss_fn(logits, y)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={running_loss/len(train_loader):.4f}, Val Loss={avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model weights
    if best_model:
        model.load_state_dict(best_model)

    # ===== Final prediction on validation set =====
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_img, X_text, _ in val_loader:
            X_img, X_text = X_img.to(device), X_text.to(device)
            logits = model(X_img, X_text)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(y_val, all_preds)
    f1 = f1_score(y_val, all_preds, average="macro")
    print(f"Final Accuracy: {acc:.4f} | F1: {f1:.4f}")

    return model, acc, f1, np.array(all_preds)
