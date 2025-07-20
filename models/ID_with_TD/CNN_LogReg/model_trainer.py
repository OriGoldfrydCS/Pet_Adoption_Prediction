"""
Model definition and training routine for CNN + Logistic Regression model.

- The CNN processes image inputs and extracts 64-dimensional features.
- TF-IDF text features are concatenated with image features.
- A single linear layer performs classification over the combined representation.
- Includes full training loop with early stopping based on validation loss.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm


# ===== Model Definition =====

class CNN_LogisticRegression(nn.Module):
    """
    CNN + Logistic Regression hybrid model.

    - CNN processes 128x128 RGB images.
    - Output image features are concatenated with TF-IDF features.
    - Combined features are fed into a linear classifier.
    """
    def __init__(self, tfidf_dim=3000, num_classes=4):
        super().__init__()

        # CNN for image input
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # output: [B, 32, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # output: [B, 32, 64, 64]
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # output: [B, 64, 64, 64]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                 # output: [B, 64, 1, 1]
        )

        # Linear layer over [image features + TF-IDF features]
        self.classifier = nn.Linear(64 + tfidf_dim, num_classes)

    def forward(self, x_img, x_text):
        img_features = self.cnn(x_img).view(x_img.size(0), -1)  # Flatten to [B, 64]
        combined = torch.cat([img_features, x_text], dim=1)     # Concatenate with TF-IDF
        return self.classifier(combined)


# ===== Training Function =====

def train_cnn_logreg(X_img_train, X_text_train, y_train,
                     X_img_val, X_text_val, y_val,
                     tfidf_dim, num_classes=4,
                     epochs=150, batch_size=64, patience=5):
    """
    Train the CNN + Logistic Regression model with early stopping.

    Args:
        - X_img_train, X_text_train, y_train: Training data (images + TF-IDF + labels)
        - X_img_val, X_text_val, y_val: Validation data
        - tfidf_dim (int): Dimensionality of TF-IDF vector
        - num_classes (int): Number of output classes
        - epochs (int): Maximum number of epochs
        - batch_size (int): Batch size
        - patience (int): Early stopping patience

    Returns:
        - model (nn.Module): Trained model
        - acc (float): Final validation accuracy
        - f1 (float): Final validation macro-F1
        - np.array: Predicted labels on validation set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert data to torch tensors
    X_img_train = torch.tensor(X_img_train, dtype=torch.float32)
    X_text_train = torch.tensor(X_text_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_img_val = torch.tensor(X_img_val, dtype=torch.float32)
    X_text_val = torch.tensor(X_text_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_img_train, X_text_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_img_val, X_text_val, y_val), batch_size=batch_size)

    # Initialize model, optimizer, loss
    model = CNN_LogisticRegression(tfidf_dim=tfidf_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_model = None
    best_loss = float('inf')
    patience_counter = 0

    # Training loop
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

        # Validation loop
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

        # Check for early stopping
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

    # Final validation predictions
    model.eval()
    all_preds = []

    with torch.no_grad():
        for X_img, X_text, _ in val_loader:
            X_img, X_text = X_img.to(device), X_text.to(device)
            logits = model(X_img, X_text)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(y_val.numpy(), all_preds)
    f1 = f1_score(y_val.numpy(), all_preds, average="macro")
    print(f"Final Accuracy: {acc:.4f} | F1: {f1:.4f}")

    return model, acc, f1, np.array(all_preds)
