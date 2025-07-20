"""
This script loads and processes a multimodal dataset consisting of:
- Structured data
- Textual description
- All images data

It uses TF-IDF to represent the text and loads all available images for each sample.
Structured data is normalized to zero mean and unit variance.
The dataset is split into train/val/test (70/15/15) and saved as a pickle file for reuse.

Note:
- Unlike other scripts that average or a few images, this version keeps the full image list per sample, becouse the tf-idf representation can leverage all available images.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

# Paths and config
CSV_PATH = "dataset/data_processed/SD_TD_ID_dataset.csv"
IMAGE_FOLDER = "dataset/original_datasets/train_images"
MAX_TFIDF_FEATURES = 3000
IMAGE_SIZE = (128, 128)
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
CACHE_PATH = "encoded_data/preprocessed_data_all_modalities_all_images.pkl"

def load_text_image_structured_data():
    # Load from cache if available
    if os.path.exists(CACHE_PATH):
        print("Loading preprocessed data from cache...")
        with open(CACHE_PATH, "rb") as f:
            data = pickle.load(f)
        return (
            (data["X_img_train"], data["X_text_train"], data["X_struct_train"], data["y_train"]),
            (data["X_img_val"], data["X_text_val"], data["X_struct_val"], data["y_val"]),
            (data["X_img_test"], data["X_text_test"], data["X_struct_test"], data["y_test"])
        )

    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    # Filter out invalid rows
    print("Filtering data...")
    df = df.dropna(subset=["Description", "image_list", "AdoptionSpeed"]).reset_index(drop=True)
    df = df[df["Description"].str.strip() != ""]
    df = df[df["image_list"].str.strip() != ""]

    # Collapse adoption speed into 4 classes
    print("Mapping labels...")
    df["label"] = df["AdoptionSpeed"].map(lambda x: 0 if x in [0, 1] else (1 if x == 2 else (2 if x == 3 else 3)))

    # TF-IDF vectorization (cached if available)
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

    print("Loading all images per sample...")
    X_img, X_text, X_struct, y = [], [], [], []

    # Pick structured numeric columns
    structured_columns = [
        col for col in df.columns
        if col not in ["Description", "image_list", "AdoptionSpeed", "label"]
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Iterate through samples
    for i, row in df.iterrows():
        img_files = row["image_list"].split(",")
        imgs = []

        # Load and resize all available images
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
            X_img.append(np.stack(imgs))  # shape: [num_images, 128, 128, 3]
            X_text.append(X_text_all[i])
            X_struct.append(row[structured_columns].values.astype(np.float32))
            y.append(row["label"])

    print(f"Successfully processed {len(X_img)} samples with images.")

    # Example shapes for sanity check
    print(f"Sample 0: number of images = {X_img[0].shape[0]}")
    print(f"Sample 1: number of images = {X_img[1].shape[0]}")
    print(f"Sample 0, image shape = {X_img[0][0].shape}")

    num_images_per_sample = [img_set.shape[0] for img_set in X_img]
    print(f"Min images per sample: {min(num_images_per_sample)}")
    print(f"Max images per sample: {max(num_images_per_sample)}")

    # Convert to numpy arrays
    X_text = np.array(X_text, dtype=np.float32)
    X_struct = np.array(X_struct, dtype=np.float32)
    y = np.array(y)

    # Normalize structured features
    X_struct = np.nan_to_num(X_struct, nan=0.0, posinf=0.0, neginf=0.0)
    X_struct_mean = X_struct.mean(axis=0)
    X_struct_std = X_struct.std(axis=0) + 1e-8
    X_struct = (X_struct - X_struct_mean) / X_struct_std

    print("After normalization → max/min:", np.max(X_struct), np.min(X_struct))

    # Split into train / val / test
    print("Splitting into train/val/test...")
    X_img_train, X_img_temp, X_text_train, X_text_temp, X_struct_train, X_struct_temp, y_train, y_temp = train_test_split(
        X_img, X_text, X_struct, y, test_size=0.3, stratify=y, random_state=42)

    X_img_val, X_img_test, X_text_val, X_text_test, X_struct_val, X_struct_test, y_val, y_test = train_test_split(
        X_img_temp, X_text_temp, X_struct_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Save to cache
    print("Saving preprocessed data to cache (via pickle)...")
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({
            "X_img_train": X_img_train,
            "X_text_train": X_text_train,
            "X_struct_train": X_struct_train,
            "y_train": y_train,
            "X_img_val": X_img_val,
            "X_text_val": X_text_val,
            "X_struct_val": X_struct_val,
            "y_val": y_val,
            "X_img_test": X_img_test,
            "X_text_test": X_text_test,
            "X_struct_test": X_struct_test,
            "y_test": y_test,
        }, f)

    return (
        (X_img_train, X_text_train, X_struct_train, y_train),
        (X_img_val, X_text_val, X_struct_val, y_val),
        (X_img_test, X_text_test, X_struct_test, y_test)
    )

# Example run
if __name__ == "__main__":
    (X_img_train, X_text_train, X_struct_train, y_train), \
    (X_img_val, X_text_val, X_struct_val, y_val), \
    (X_img_test, X_text_test, X_struct_test, y_test) = load_text_image_structured_data()

    print("\nData loaded successfully!")
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"Text shape: {X_text_train.shape[1]}, Structured shape: {X_struct_train.shape[1]}")
    print(f"Train sample 0: image count = {X_img_train[0].shape[0]}, single image shape = {X_img_train[0][0].shape}")
