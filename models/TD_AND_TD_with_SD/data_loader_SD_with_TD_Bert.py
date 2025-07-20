"""
Data Loader: BERT + Structured Features

This script prepares a dataset that combines:
1. Text embeddings (from BERT) for two fields: Description and Name
2. Structured data (numerical and categorical)
3. Target labels for a 4-class classification task

The script performs:
- Text cleaning and filtering
- Tokenization and embedding extraction using BERT (CLS + token-level)
- Label encoding of structured features
- Standardization and imputation
- Merging text and structured features
- Train/val/test split (70%/15%/15%) before any transformations

Returns a dictionary with:
- Combined CLS + structured feature matrix (NumPy)
- Token-level BERT embeddings for each text field (PyTorch tensors)
- Train, val, and test labels
"""

import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# Constants
DATA_PATH = "dataset/data_processed/data/SD_TD_dataset.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(DEVICE)
bert_model.eval()


def get_bert_cls_and_tokens(texts, batch_size=32, max_length=128, label=""):
    """
    Tokenizes input texts and extracts both CLS and token-level embeddings using BERT.

    Args:
        - texts (List or Series): Input text data
        - batch_size (int): Number of samples per batch
        - max_length (int): Max token length for BERT
        - label (str): Optional label for tqdm logging

    Returns:
        - tuple: (CLS embeddings [N, 768], Token embeddings [N, seq_len, 768])
    """
    print(f"Extracting BERT CLS + token embeddings for {label}...")
    cls_outputs, token_outputs = [], []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=label):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(
                batch.tolist(),
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(DEVICE)

            output = bert_model(**encoded)
            last_hidden_state = output.last_hidden_state
            cls = last_hidden_state[:, 0, :]  # CLS token

            cls_outputs.append(cls.cpu())
            token_outputs.append(last_hidden_state.cpu())

    return (
        torch.cat(cls_outputs, dim=0),       # [N, 768]
        torch.cat(token_outputs, dim=0)      # [N, seq_len, 768]
    )


def load_train_val_test_with_bert_and_structured():
    """
    Loads the dataset, splits it, extracts BERT features for Description and Name,
    and combines them with standardized structured features.

    Returns dictionary with:
        - "X_cls_combined": CLS embeddings + structured features (NumPy arrays)
        - "X_token_desc": Token embeddings for Description (tensors)
        - "X_token_name": Token embeddings for Name (tensors)
        - "y": Labels split into train, val, test
    """
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["AdoptionSpeed", "Description", "Name"])
    df = df[(df["Description"].str.strip() != "") & (df["Name"].str.strip() != "")]

    # Label mapping: [0,1] → 0, 2 → 1, 3 → 2, 4 → 3
    df["label"] = df["AdoptionSpeed"].map(lambda x: 0 if x in [0, 1] else (1 if x == 2 else (2 if x == 3 else 3)))
    y_all = df["label"].reset_index(drop=True)

    # Raw text fields
    desc_texts_all = df["Description"].astype(str).reset_index(drop=True)
    name_texts_all = df["Name"].astype(str).reset_index(drop=True)

    # Split before encoding (important!)
    all_indices = np.arange(len(y_all))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        all_indices, y_all, test_size=0.3, stratify=y_all, random_state=42)
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Prepare text splits
    desc_train_texts = desc_texts_all[train_idx].reset_index(drop=True)
    desc_val_texts   = desc_texts_all[val_idx].reset_index(drop=True)
    desc_test_texts  = desc_texts_all[test_idx].reset_index(drop=True)

    name_train_texts = name_texts_all[train_idx].reset_index(drop=True)
    name_val_texts   = name_texts_all[val_idx].reset_index(drop=True)
    name_test_texts  = name_texts_all[test_idx].reset_index(drop=True)

    # Encode text (BERT CLS + tokens)
    desc_cls_train, desc_tokens_train = get_bert_cls_and_tokens(desc_train_texts, label="Desc Train")
    desc_cls_val,   desc_tokens_val   = get_bert_cls_and_tokens(desc_val_texts, label="Desc Val")
    desc_cls_test,  desc_tokens_test  = get_bert_cls_and_tokens(desc_test_texts, label="Desc Test")

    name_cls_train, name_tokens_train = get_bert_cls_and_tokens(name_train_texts, label="Name Train")
    name_cls_val,   name_tokens_val   = get_bert_cls_and_tokens(name_val_texts, label="Name Val")
    name_cls_test,  name_tokens_test  = get_bert_cls_and_tokens(name_test_texts, label="Name Test")

    # Process structured features
    drop_cols = ["PetID", "RescuerID", "VideoAmt", "PhotoAmt", "AdoptionSpeed", "Description", "Name"]
    structured_df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Encode categorical columns
    for col in structured_df.select_dtypes(include=["object"]).columns:
        structured_df[col] = LabelEncoder().fit_transform(structured_df[col].astype(str))

    # Handle missing values
    X_structured_all = structured_df.values.astype(np.float32)
    X_structured_all = SimpleImputer(strategy="constant", fill_value=0).fit_transform(X_structured_all)

    # Standardize using only training data
    scaler = StandardScaler()
    X_structured_train = scaler.fit_transform(X_structured_all[train_idx])
    X_structured_val   = scaler.transform(X_structured_all[val_idx])
    X_structured_test  = scaler.transform(X_structured_all[test_idx])

    # Combine CLS vectors and structured features
    X_cls_combined_train = np.hstack([desc_cls_train.numpy(), name_cls_train.numpy(), X_structured_train])
    X_cls_combined_val   = np.hstack([desc_cls_val.numpy(),   name_cls_val.numpy(),   X_structured_val])
    X_cls_combined_test  = np.hstack([desc_cls_test.numpy(),  name_cls_test.numpy(),  X_structured_test])

    return {
        "X_cls_combined": (
            X_cls_combined_train,
            X_cls_combined_val,
            X_cls_combined_test
        ),
        "X_token_desc": (
            desc_tokens_train,
            desc_tokens_val,
            desc_tokens_test
        ),
        "X_token_name": (
            name_tokens_train,
            name_tokens_val,
            name_tokens_test
        ),
        "y": (
            y_train.reset_index(drop=True),
            y_val.reset_index(drop=True),
            y_test.reset_index(drop=True)
        )
    }

# Run and preview output shapes
if __name__ == "__main__":
    data = load_train_val_test_with_bert_and_structured()
    print("CLS+structured Train shape:", data["X_cls_combined"][0].shape)
    print("Token Description shape:", data["X_token_desc"][0].shape)
    print("Token Name shape:", data["X_token_name"][0].shape)
