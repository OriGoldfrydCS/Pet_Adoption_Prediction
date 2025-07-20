"""
This script loads and preprocesses a multimodal dataset containing:
- TF-IDF encoded text descriptions
- Structured data
- Images per sample

Unlike other loaders, this version **averages all images per sample** into a single image
before training, to reduce memory usage and simplify modeling.

Main steps:
1. Loads dataset from CSV
2. Extracts TF-IDF features (with caching)
3. Loads and averages image sequences
4. Normalizes structured features
5. Splits data into train / val / test
6. Saves results to compressed `.npz` cache file
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

# ==== Configuration ====
CSV_PATH = "dataset/data_processed/SD_TD_ID_dataset.csv"
IMAGE_FOLDER = "dataset/original_datasets/train_images"
MAX_TFIDF_FEATURES = 3000
IMAGE_SIZE = (128, 128)
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
CACHE_PATH = "encoded_data/preprocessed_data_all_modalities.npz"


def load_text_image_structured_data():
    """
    Loads the dataset, applies preprocessing (TF-IDF, structured normalization),
    and averages all images per sample. Returns train/val/test splits.
    """
    # Load from cache if available
    if os.path.exists(CACHE_PATH):
        print("Loading preprocessed data from cache...")
        data = np.load(CACHE_PATH)
        return (
            (data["X_img_train"], data["X_text_train"], data["X_struct_train"], data["y_train"]),
            (data["X_img_val"], data["X_text_val"], data["X_struct_val"], data["y_val"]),
            (data["X_img_test"], data["X_text_test"], data["X_struct_test"], data["y_test"])
        )

    # === Load and clean raw CSV ===
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    print("Filtering data...")
    df = df.dropna(subset=["Description", "image_list", "AdoptionSpeed"]).reset_index(drop=True)
    df = df[df["Description"].str.strip() != ""]
    df = df[df["image_list"].str.strip() != ""]

    print("Mapping labels...")
    df["label"] = df["AdoptionSpeed"].map(lambda x: 0 if x in [0, 1] else (1 if x == 2 else (2 if x == 3 else 3)))

    # === TF-IDF Vectorization ===
    print("Extracting TF-IDF features...")
    if os.path.exists(VECTORIZER_PATH):
        print("Loading cached TF-IDF vectorizer...")
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
    else:
        print("Fitting new TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(vectorizer, f)

    X_text_all = vectorizer.fit_transform(df["Description"]).toarray()

    # === Load and average images ===
    print("Loading and averaging all images per sample...")
    X_img, X_text, X_struct, y = [], [], [], []

    structured_columns = [
        col for col in df.columns
        if col not in ["Description", "image_list", "AdoptionSpeed", "label"]
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    for i, row in df.iterrows():
        img_files = row["image_list"].split(",")
        imgs = []

        for file in img_files:
            img_path = os.path.join(IMAGE_FOLDER, file.strip())
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
                    imgs.append(np.array(img, dtype=np.float32) / 255.0)
                except Exception as e:
                    print(f"Failed to load image: {img_path} — {e}")
                    continue

        if len(imgs) > 0:
            avg_img = np.mean(imgs, axis=0)  # Averaging images into one
            X_img.append(avg_img)
            X_text.append(X_text_all[i])
            X_struct.append(row[structured_columns].values.astype(np.float32))
            y.append(row["label"])

    print(f"Successfully processed {len(X_img)} samples with images.")

    # === Convert to numpy arrays ===
    X_img = np.array(X_img, dtype=np.float32)
    X_text = np.array(X_text, dtype=np.float32)
    X_struct = np.array(X_struct, dtype=np.float32)
    y = np.array(y)

    # === Normalize structured features ===
    X_struct = np.nan_to_num(X_struct, nan=0.0, posinf=0.0, neginf=0.0)
    X_struct_mean = X_struct.mean(axis=0)
    X_struct_std = X_struct.std(axis=0) + 1e-8
    X_struct = (X_struct - X_struct_mean) / X_struct_std

    print("After normalization → max/min:", np.max(X_struct), np.min(X_struct))

    # === Train/val/test split ===
    print("Splitting into train/val/test...")
    X_img_train, X_img_temp, X_text_train, X_text_temp, X_struct_train, X_struct_temp, y_train, y_temp = train_test_split(
        X_img, X_text, X_struct, y, test_size=0.3, stratify=y, random_state=42)

    X_img_val, X_img_test, X_text_val, X_text_test, X_struct_val, X_struct_test, y_val, y_test = train_test_split(
        X_img_temp, X_text_temp, X_struct_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # === Save to compressed cache ===
    print("Saving preprocessed data to cache...")
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        X_img_train=X_img_train, X_text_train=X_text_train, X_struct_train=X_struct_train, y_train=y_train,
        X_img_val=X_img_val, X_text_val=X_text_val, X_struct_val=X_struct_val, y_val=y_val,
        X_img_test=X_img_test, X_text_test=X_text_test, X_struct_test=X_struct_test, y_test=y_test
    )

    return (
        (X_img_train, X_text_train, X_struct_train, y_train),
        (X_img_val, X_text_val, X_struct_val, y_val),
        (X_img_test, X_text_test, X_struct_test, y_test)
    )


# ==== Run as script (sanity check) ====
if __name__ == "__main__":
    (X_img_train, X_text_train, X_struct_train, y_train), \
    (X_img_val, X_text_val, X_struct_val, y_val), \
    (X_img_test, X_text_test, X_struct_test, y_test) = load_text_image_structured_data()

    print("\nData loaded successfully!")
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"Image shape: {X_img_train.shape[1:]}, Text shape: {X_text_train.shape[1]}, Structured shape: {X_struct_train.shape[1]}")
