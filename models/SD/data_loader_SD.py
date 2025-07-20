# This script loads structured data from CSV files, cleans it, encodes categorical features, maps the target labels, and returns train(70%)/val(15%)/test(15%) splits.

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# Folder with pre-split CSVs for train/val/test
DATA_DIR = "dataset/splits"

def load_basic_features(file_path):
    """
    Loads a dataset CSV and prepares it for modeling.
    Drops irrelevant columns, encodes categoricals, and maps labels.
    """
    df = pd.read_csv(file_path)

    # Remove IDs, names, descriptions, and videos – we won't use them
    drop_cols = ["PetID", "Name", "RescuerID", "Description", "VideoAmt"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Get rid of rows with missing values
    df = df.dropna()

    # Encode categorical columns with LabelEncoder
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Merge classes 0 and 1 → new class 0
    y = df["AdoptionSpeed"].map(lambda x: 0 if x in [0, 1] else x)
    y = LabelEncoder().fit_transform(y)

    # Drop the label column from features
    X = df.drop("AdoptionSpeed", axis=1)

    return X, y

def load_train_val_test():
    """
    Loads and returns train, val, and test splits.
    """
    X_train, y_train = load_basic_features(os.path.join(DATA_DIR, "train_SD_TD_dataset.csv"))
    X_val, y_val = load_basic_features(os.path.join(DATA_DIR, "val_SD_TD_dataset.csv"))
    X_test, y_test = load_basic_features(os.path.join(DATA_DIR, "test_SD_TD_dataset.csv"))
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

if __name__ == "__main__":
    # Quick check to verify shapes
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test()
    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)
