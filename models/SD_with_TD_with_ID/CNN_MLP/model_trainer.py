"""
CNN + MLP model for multimodal classification:
- CNN for image input (average image per sample)
- MLP for TF-IDF text features
- MLP for structured numeric features

This script defines the model architecture and the training + evaluation logic.
The best model is selected based on validation loss with early stopping.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm

# Define the multimodal neural network
class CNN_MLP(nn.Module):
    def __init__(self, num_classes=4, text_dim=3000, struct_dim=15, dropout=0.3):
        super().__init__()

        # CNN for image input
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Output shape: [batch_size, 128, 1, 1]
        )

        # MLP for text (TF-IDF)
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # MLP for structured features
        self.struct_branch = nn.Sequential(
            nn.Linear(struct_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Fusion of all branches and classification
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128 + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_text, x_struct):
        img_features = self.cnn(x_img).view(x_img.size(0), -1)
        text_features = self.text_branch(x_text)
        struct_features = self.struct_branch(x_struct)
        x = torch.cat([img_features, text_features, struct_features], dim=1)
        return self.classifier(x)


# Training and evaluation logic
def train_and_evaluate(X_img_train, X_text_train, X_struct_train, y_train,
                       X_img_val, X_text_val, X_struct_val, y_val,
                       epochs=50, batch_size=64, patience=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert numpy arrays to PyTorch tensors
    X_img_train = torch.tensor(X_img_train, dtype=torch.float32)
    X_text_train = torch.tensor(X_text_train, dtype=torch.float32)
    X_struct_train = torch.tensor(X_struct_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_img_val = torch.tensor(X_img_val, dtype=torch.float32)
    X_text_val = torch.tensor(X_text_val, dtype=torch.float32)
    X_struct_val = torch.tensor(X_struct_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Create PyTorch datasets and dataloaders
    train_loader = DataLoader(TensorDataset(X_img_train, X_text_train, X_struct_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_img_val, X_text_val, X_struct_val, y_val),
                            batch_size=batch_size)

    # Initialize the model
    model = CNN_MLP(
        num_classes=4,
        text_dim=X_text_train.shape[1],
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

        for X_img, X_text, X_struct, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
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

        # Early stopping logic
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

    # Final predictions on validation set
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_img, X_text, X_struct, _ in val_loader:
            X_img, X_text, X_struct = X_img.to(device), X_text.to(device), X_struct.to(device)
            logits = model(X_img, X_text, X_struct)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    # Final evaluation metrics
    acc = accuracy_score(y_val.cpu().numpy(), all_preds)
    f1 = f1_score(y_val.cpu().numpy(), all_preds, average="macro")
    print(f"Validation Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    return model, np.array(all_preds)
