"""
Multimodal Data Loader (Images + Text + Structured)

This script loads and processes a dataset that includes:
- Structured features (normalized numerics)
- Text descriptions (embedded via RoBERTa, shape: [128, 768])
- Multiple images per sample (resized, normalized, padded to MAX_IMAGES)

Main steps:
1. Load data from CSV.
2. Filter invalid rows.
3. Process each modality (images, text, structured).
4. Pad/truncate to fixed image count (MAX_IMAGES).
5. Normalize structured features.
6. Split into train/val/test sets.
7. Cache the output as a .pkl file.

Returns:
Train/Val/Test sets for all 3 modalities and labels.
"""

# Load required libraries
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

# File paths and config
CSV_PATH = "dataset/data_processed/SD_TD_ID_dataset.csv"
IMAGE_FOLDER = "dataset/original_datasets/train_images"
IMAGE_SIZE = (128, 128)
CACHE_PATH = "encoded_data/preprocessed_data_roberta_all_modalities_5_images.pkl"
MAX_IMAGES = 5  # Number of images to use per sample (pad/truncate as needed)

# Load RoBERTa tokenizer and model
print("Loading RoBERTa model...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base")
roberta_model.eval()
roberta_model = roberta_model.cuda() if torch.cuda.is_available() else roberta_model.cpu()


def load_text_image_structured_data():
    # If cache exists, just load from it
    if os.path.exists(CACHE_PATH):
        print("Loading preprocessed data from cache...")
        with open(CACHE_PATH, "rb") as f:
            data = pickle.load(f)
            return (
                (data["X_img_train"], data["X_text_train"], data["X_struct_train"], data["y_train"]),
                (data["X_img_val"], data["X_text_val"], data["X_struct_val"], data["y_val"]),
                (data["X_img_test"], data["X_text_test"], data["X_struct_test"], data["y_test"])
            )

    # Load raw data
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["Description", "image_list", "AdoptionSpeed"]).reset_index(drop=True)
    df = df[df["Description"].str.strip() != ""]
    df = df[df["image_list"].str.strip() != ""]

    # Map labels to 4 classes
    print("Mapping labels...")
    df["label"] = df["AdoptionSpeed"].map(lambda x: 0 if x in [0, 1] else (1 if x == 2 else (2 if x == 3 else 3)))

    print("Processing images, text embeddings, and structured data...")
    X_img, X_text, X_struct, y = [], [], [], []
    structured_columns = [
        col for col in df.columns
        if col not in ["Description", "image_list", "AdoptionSpeed", "label"]
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_files = row["image_list"].split(",")
        imgs = []

        # Load and resize up to MAX_IMAGES
        for file in img_files:
            img_path = os.path.join(IMAGE_FOLDER, file.strip())
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
                    imgs.append(np.array(img, dtype=np.float32) / 255.0)
                except:
                    continue
            if len(imgs) == MAX_IMAGES:
                break

        # Pad or trim to fixed length
        if len(imgs) < MAX_IMAGES:
            pad = [np.zeros((128, 128, 3), dtype=np.float32)] * (MAX_IMAGES - len(imgs))
            imgs.extend(pad)
        elif len(imgs) > MAX_IMAGES:
            imgs = imgs[:MAX_IMAGES]

        if len(imgs) != MAX_IMAGES:
            print(f"[Skipping] Sample {i} has {len(imgs)} images after adjustment")
            continue

        try:
            # Encode text using RoBERTa
            desc = row["Description"]
            encoding = tokenizer(
                desc,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(roberta_model.device)
            with torch.no_grad():
                output = roberta_model(**encoding)
                emb = output.last_hidden_state.squeeze(0).cpu().numpy()  # shape: (128, 768)

            # Store everything
            X_img.append(np.stack(imgs))  # shape: (MAX_IMAGES, 128, 128, 3)
            X_text.append(emb)            # shape: (128, 768)
            X_struct.append(row[structured_columns].values.astype(np.float32))
            y.append(row["label"])
        except Exception as e:
            print(f"[Error] Sample {i}: {e}")

    print(f"Successfully processed {len(X_img)} samples with {MAX_IMAGES} images per sample.")

    # Normalize structured features
    X_struct = np.array(X_struct)
    X_struct = np.nan_to_num(X_struct)
    X_struct = (X_struct - X_struct.mean(axis=0)) / (X_struct.std(axis=0) + 1e-8)
    y = np.array(y)

    # Split into train/val/test
    X_img_train, X_img_temp, X_text_train, X_text_temp, X_struct_train, X_struct_temp, y_train, y_temp = train_test_split(
        X_img, X_text, X_struct, y, test_size=0.3, stratify=y, random_state=42)
    X_img_val, X_img_test, X_text_val, X_text_test, X_struct_val, X_struct_test, y_val, y_test = train_test_split(
        X_img_temp, X_text_temp, X_struct_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Save processed data
    data = {
        "X_img_train": np.array(X_img_train),
        "X_text_train": np.array(X_text_train),
        "X_struct_train": X_struct_train,
        "y_train": y_train,
        "X_img_val": np.array(X_img_val),
        "X_text_val": np.array(X_text_val),
        "X_struct_val": X_struct_val,
        "y_val": y_val,
        "X_img_test": np.array(X_img_test),
        "X_text_test": np.array(X_text_test),
        "X_struct_test": X_struct_test,
        "y_test": y_test,
    }

    print("Saving to cache...")
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(data, f)

    return (
        (data["X_img_train"], data["X_text_train"], data["X_struct_train"], data["y_train"]),
        (data["X_img_val"], data["X_text_val"], data["X_struct_val"], data["y_val"]),
        (data["X_img_test"], data["X_text_test"], data["X_struct_test"], data["y_test"])
    )


# Run function if file is executed directly
if __name__ == "__main__":
    (X_img_train, X_text_train, X_struct_train, y_train), \
    (X_img_val, X_text_val, X_struct_val, y_val), \
    (X_img_test, X_text_test, X_struct_test, y_test) = load_text_image_structured_data()

    print("\nData loaded successfully!")
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"Text shape: {X_text_train.shape}, Structured: {X_struct_train.shape}")
    print(f"Example image shape: {X_img_train[0].shape}")
    print("\nFinal sanity check (train split):")
    print(f"  Total train samples: {len(X_img_train)}")
    print(f"  Image[0] shape:      {X_img_train[0].shape}")
    print(f"  Text shape:          {X_text_train.shape}")
    print(f"  Struct shape:        {X_struct_train.shape}")
    print(f"  Labels shape:        {y_train.shape}")
    print(f"  Unique labels:       {np.unique(y_train)}")
