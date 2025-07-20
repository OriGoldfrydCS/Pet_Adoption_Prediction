"""
TF-IDF + Structured data loader

This script loads a dataset containing both free-text fields (Description, Name)
and structured data. It performs the following steps:

1. Cleans the data and encodes the target variable (AdoptionSpeed) into 4 classes.
2. Extracts TF-IDF features from selected text columns (Description + Name).
3. Encodes categorical structured features using label encoding.
4. Concatenates text and structured features into a single matrix.
5. Applies a train/val/test split (70% / 15% / 15%).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

DATA_PATH = "dataset/data_processed/data/SD_TD_dataset.csv"
TEXT_COLS = ["Description", "Name"]
MAX_FEATURES = 1000

def load_train_val_test_with_text(four_class=True):
    """
    Loads the dataset, applies TF-IDF on text fields, encodes structured features,
    and returns combined feature matrices with 4-class labels.

    Args:
        - four_class (bool): Whether to apply 4-class mapping on labels.

    Returns:
        Tuple of splits:
        - (X_train, y_train)
        - (X_val, y_val)
        - (X_test, y_test)
    """
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["AdoptionSpeed"])

    # Label mapping (e.g., 0+1 → 0, then 2, 3, 4 remain distinct)
    if four_class:
        df["label"] = df["AdoptionSpeed"].map(lambda x: 0 if x in [0, 1] else x)
        df["label"] = pd.factorize(df["label"])[0]
    else:
        df["label"] = df["AdoptionSpeed"]

    # Encode TF-IDF for text columns
    text_matrix = []
    for col in TEXT_COLS:
        tfidf = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english")
        tfidf_matrix = tfidf.fit_transform(df[col].fillna(""))
        text_matrix.append(tfidf_matrix)

    # Select and encode structured features
    drop_cols = ["PetID", "RescuerID", "VideoAmt", "PhotoAmt", "AdoptionSpeed"] + TEXT_COLS
    structured_df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Label encode any remaining categorical columns
    for col in structured_df.select_dtypes(include=["object"]).columns:
        structured_df[col] = LabelEncoder().fit_transform(structured_df[col].astype(str))

    X_structured = structured_df.values

    # Combine structured and text features into one feature matrix
    X = hstack([X_structured] + text_matrix)

    # Fill any missing values with 0
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X = imputer.fit_transform(X.toarray())  # Convert sparse matrix to dense before imputation

    y = df["label"]

    # Split: 70% train / 15% val / 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Run and display shapes
if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test_with_text(four_class=True)
    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)
