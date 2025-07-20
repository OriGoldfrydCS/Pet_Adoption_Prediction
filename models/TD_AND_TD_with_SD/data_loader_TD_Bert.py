"""
BERT-Based Text Embedding and Dataset Preparation

This script loads a dataset with text descriptions and adoption speed labels,
cleans and splits it into train(70%)/val(15%)/test(15%) sets, and encodes the text using BERT.

For each sample, it extracts three types of embeddings:
1. CLS token embedding (used often for classification)
2. Mean-pooled embedding across all tokens (ignores padding)
3. Full token-level embedding (sequence of hidden states)
"""

import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
bert_model.to(device)
bert_model.eval()

DATA_PATH = "dataset/data_processed/data/SD_TD_dataset.csv"

def get_bert_embeddings(texts, batch_size=32, max_length=128, label=""):
    """
    Generate BERT embeddings (CLS, mean, token-level) for a batch of texts.

    Args:
        - texts (List or Series): Input sentences.
        - batch_size (int): Batch size for BERT inference.
        - max_length (int): Max token length per sentence.
        - label (str): Label for logging.

    Returns:
        tuple of tensors:
            - CLS embeddings [N, 768]
            - Mean pooled embeddings [N, 768]
            - Token-level embeddings [N, seq_len, 768]
    """
    print(f"Extracting BERT embeddings for {label} set...")
    cls_outputs = []
    token_outputs = []
    mean_outputs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Processing {label}"):
            batch = texts[i:i+batch_size]

            # Tokenize and move to device
            encoded = tokenizer(
                batch.tolist(),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            # Forward pass through BERT
            output = bert_model(**encoded)
            last_hidden_state = output.last_hidden_state  # [B, seq_len, 768]
            cls_embeddings = last_hidden_state[:, 0, :]   # CLS token [B, 768]

            # Mean pooling (ignores padding)
            attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
            sum_mask = attention_mask.sum(dim=1)
            mean_pooled = sum_embeddings / sum_mask  # [B, 768]

            # Store outputs
            cls_outputs.append(cls_embeddings.cpu())
            token_outputs.append(last_hidden_state.cpu())
            mean_outputs.append(mean_pooled.cpu())

    # Concatenate batches
    cls_tensor = torch.cat(cls_outputs, dim=0)
    token_tensor = torch.cat(token_outputs, dim=0)
    mean_tensor = torch.cat(mean_outputs, dim=0)
    return cls_tensor, mean_tensor, token_tensor


def load_train_val_test_with_bert():
    """
    Loads the dataset, applies label mapping, splits into train/val/test,
    and computes BERT embeddings for each split.

    Returns:
        Tuple of tuples:
        - (train_cls, train_mean, train_tokens, y_train)
        - (val_cls, val_mean, val_tokens, y_val)
        - (test_cls, test_mean, test_tokens, y_test)
    """
    # Load and clean dataset
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Description", "AdoptionSpeed"])
    df = df[df["Description"].str.strip() != ""]

    # Map 5 adoption speeds to 4 classes: [0,1]→0, 2→1, 3→2, 4→3
    df["label"] = df["AdoptionSpeed"].map(
        lambda x: 0 if x in [0, 1] else (1 if x == 2 else (2 if x == 3 else 3))
    )

    X_text = df["Description"]
    y = df["label"]

    # Split into train (70%), val (15%), test (15%)
    X_train_val_text, X_test_text, y_train_val, y_test = train_test_split(
        X_text, y, test_size=0.15, stratify=y, random_state=42
    )

    val_size = 0.15 / 0.85
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_train_val_text, y_train_val, test_size=val_size, stratify=y_train_val, random_state=42
    )

    # Extract BERT features for each split
    X_train_cls, X_train_mean, X_train_tokens = get_bert_embeddings(X_train_text, label="Train")
    X_val_cls, X_val_mean, X_val_tokens = get_bert_embeddings(X_val_text, label="Validation")
    X_test_cls, X_test_mean, X_test_tokens = get_bert_embeddings(X_test_text, label="Test")

    return (
        (X_train_cls, X_train_mean, X_train_tokens, y_train.reset_index(drop=True)),
        (X_val_cls, X_val_mean, X_val_tokens, y_val.reset_index(drop=True)),
        (X_test_cls, X_test_mean, X_test_tokens, y_test.reset_index(drop=True))
    )

# ========== Entry Point ========== #
if __name__ == "__main__":
    (X_train_cls, X_train_mean, X_train_tokens, y_train), \
    (X_val_cls, X_val_mean, X_val_tokens, y_val), \
    (X_test_cls, X_test_mean, X_test_tokens, y_test) = load_train_val_test_with_bert()
    
    print("Finished BERT encoding.")
    print("Train CLS shape:", X_train_cls.shape)
    print("Train Mean shape:", X_train_mean.shape)
    print("Train Tokens shape:", X_train_tokens.shape)
    print("Val CLS shape:", X_val_cls.shape)
    print("Val Mean shape:", X_val_mean.shape)
    print("Val Tokens shape:", X_val_tokens.shape)
    print("Test CLS shape:", X_test_cls.shape)
    print("Test Mean shape:", X_test_mean.shape)
    print("Test Tokens shape:", X_test_tokens.shape)
