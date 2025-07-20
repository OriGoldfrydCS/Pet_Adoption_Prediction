"""
TF-IDF Text Preprocessing and Dataset Splitting

This script loads a dataset containing free-text descriptions and corresponding adoption labels.
It performs the following:
1. Cleans the data (drops missing/empty text).
2. Converts the 'AdoptionSpeed' column into a classification label (4-class).
3. Splits the dataset into training, validation, and test sets (70% / 15% / 15%).
4. Applies TF-IDF vectorization to the text (optionally limiting max features).

Returns train/val/test datasets ready for traditional ML models.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = "dataset/data_processed/data/SD_TD_dataset.csv"

def load_train_val_test_with_text(binary=False, three_class=True, four_class=False, max_features=3000):
    """
    Loads text descriptions and returns TF-IDF-transformed train/val/test sets.

    Args:
        - binary (bool): If True, maps labels to 0/1 based on fast vs. slow adoption.
        - four_class (bool): If True, maps to 4 classes: [0,1]→0, then keeps 2,3,4.
        - max_features (int): Max number of TF-IDF features.

    Returns:
        Tuple:
        - (X_train, y_train)
        - (X_val, y_val)
        - (X_test, y_test)
    """
    # Load and clean data
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Description", "AdoptionSpeed"])
    df = df[df["Description"].str.strip() != ""]

    # Label mapping
    if binary:
        df["label"] = df["AdoptionSpeed"].map(lambda x: 0 if x in [0, 1] else 1)
    elif three_class:
        df["label"] = df["AdoptionSpeed"].map(lambda x: 0 if x in [0, 1] else (1 if x == 2 else 2))
    elif four_class:
        df["label"] = df["AdoptionSpeed"].map(lambda x: 0 if x in [0, 1] else x)
        df["label"] = pd.factorize(df["label"])[0]  # Ensure numeric labels
    else:
        df["label"] = df["AdoptionSpeed"]

    # Extract features and targets
    X_text = df["Description"]
    y = df["label"]

    # Split: 85% train+val, 15% test
    X_train_val_text, X_test_text, y_train_val, y_test = train_test_split(
        X_text, y, test_size=0.15, stratify=y, random_state=42
    )

    # From 85%, reserve ~17.6% (≈15% of total) for validation
    val_size = 0.15 / 0.85
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_train_val_text, y_train_val, test_size=val_size, stratify=y_train_val, random_state=42
    )

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# Run & print shapes (for quick check)
if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test_with_text()
    print("Train shape:", X_train.shape)
    print("Val shape:", X_val.shape)
    print("Test shape:", X_test.shape)
