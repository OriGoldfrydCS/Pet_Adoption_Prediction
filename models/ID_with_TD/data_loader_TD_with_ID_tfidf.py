# This script loads structured data with image and text fields, extracts TF-IDF features from descriptions, averages multiple images per sample, and returns train/val/test splits. Saves the TF-IDF vectorizer for later use.

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

# === Constants ===
CSV_PATH = "dataset/data_processed/SD_TD_ID_dataset.csv"
IMAGE_FOLDER = "dataset/original_datasets/train_images"
MAX_TFIDF_FEATURES = 3000  # TF-IDF vocabulary size
IMAGE_SIZE = (128, 128)  # image resize shape


def load_text_image_data():
    """
    Loads and preprocesses text and image data.

    Returns:
        train/val/test splits for image array, TF-IDF features, and labels.
    """
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    # Remove rows with missing or empty fields
    print("Filtering data...")
    df = df.dropna(subset=["Description", "image_list", "AdoptionSpeed"]).reset_index(drop=True)
    df = df[df["Description"].str.strip() != ""]
    df = df[df["image_list"].str.strip() != ""]

    # Map AdoptionSpeed (0-4) → 4-class label
    print("Mapping labels...")
    df["label"] = df["AdoptionSpeed"].map(
        lambda x: 0 if x in [0, 1] else (1 if x == 2 else (2 if x == 3 else 3))
    )

    # Extract text features using TF-IDF
    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES)
    X_text_all = vectorizer.fit_transform(df["Description"]).toarray()

    # Save vectorizer for future inference
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Process and average all images per sample
    print("Loading and averaging all images per sample...")
    X_img, X_text, y = [], [], []

    for i, row in df.iterrows():
        img_files = row["image_list"].split(",")  # list of image file names
        imgs = []

        for file in img_files:
            img_path = os.path.join(IMAGE_FOLDER, file.strip())
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
                    imgs.append(np.array(img, dtype=np.float32) / 255.0)  # normalize to [0, 1]
                except Exception as e:
                    print(f"Failed to load image: {img_path} — {e}")
                    continue

        # If we have valid images, average them and save features
        if len(imgs) > 0:
            avg_img = np.mean(imgs, axis=0)
            X_img.append(avg_img)
            X_text.append(X_text_all[i])
            y.append(row["label"])

    print(f"Successfully processed {len(X_img)} samples with images.")

    # Convert lists to numpy arrays
    X_img = np.array(X_img, dtype=np.float32)
    X_text = np.array(X_text, dtype=np.float32)
    y = np.array(y)

    # Split into train/val/test sets
    print("Splitting into train/val/test...")
    X_img_train, X_img_temp, X_text_train, X_text_temp, y_train, y_temp = train_test_split(
        X_img, X_text, y, test_size=0.3, stratify=y, random_state=42)

    X_img_val, X_img_test, X_text_val, X_text_test, y_val, y_test = train_test_split(
        X_img_temp, X_text_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    return (
        (X_img_train, X_text_train, y_train),
        (X_img_val, X_text_val, y_val),
        (X_img_test, X_text_test, y_test)
    )


# Run preprocessing if executed directly
if __name__ == "__main__":
    (X_img_train, X_text_train, y_train), \
    (X_img_val, X_text_val, y_val), \
    (X_img_test, X_text_test, y_test) = load_text_image_data()

    print("\nData loaded successfully!")
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"Image shape: {X_img_train.shape[1:]}, Text shape: {X_text_train.shape[1]}")