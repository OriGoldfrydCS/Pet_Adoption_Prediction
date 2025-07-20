"""
This script loads and processes a multimodal dataset consisting of:
- Structured numeric features
- Textual descriptions
- Up to MAX_IMAGES per sample

It uses the CLIP model to extract embeddings from both images and text.
The structured features are normalized.
The dataset is split into train/val/test (70/15/15) and cached as a pickle file.

Configuration:
- Set `USE_IMAGE_MEAN = True` to average image embeddings into a single vector.
- Set `USE_IMAGE_MEAN = False` to retain a sequence of image embeddings per sample.

Notre:
- we tried to use more than 10 images per sample, but it was too heavy for memory
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle
import torch
from tqdm import tqdm
import open_clip

# === Configuration Constants ===
CSV_PATH = "dataset/data_processed/SD_TD_ID_dataset.csv"
IMAGE_FOLDER = "dataset/original_datasets/train_images"
CACHE_PATH = "encoded_data/preprocessed_data_clip_all_modalities_10_images_not_mean.pkl"
MAX_IMAGES = 10 # Max number of images to use for each sample (so it fits in memory)
USE_IMAGE_MEAN = False  # Set to True to use the mean of the MAX_IMAGES images per sample

# === Load CLIP model and tokenizer ===
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model.eval().to(device)


def load_text_image_structured_data():
    """
    Loads and processes the dataset. If a preprocessed cache exists, it will be loaded.
    Otherwise, data will be processed and saved to cache.

    Returns:
        Tuple of train, val, and test splits:
            (X_img_train, X_text_train, X_struct_train, y_train),
            (X_img_val, X_text_val, X_struct_val, y_val),
            (X_img_test, X_text_test, X_struct_test, y_test)
    """
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

    # === Step 1: Load CSV and clean ===
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["Description", "image_list", "AdoptionSpeed"]).reset_index(drop=True)
    df = df[df["Description"].str.strip() != ""]
    df = df[df["image_list"].str.strip() != ""]

    print("Mapping labels...")
    # Map AdoptionSpeed to 4-class labels
    df["label"] = df["AdoptionSpeed"].map(lambda x: 0 if x in [0, 1] else (1 if x == 2 else (2 if x == 3 else 3)))

    print("Processing images, text embeddings, and structured data...")
    X_img, X_text, X_struct, y = [], [], [], []
    structured_columns = [
        col for col in df.columns 
        if col not in ["Description", "image_list", "AdoptionSpeed", "label"] 
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    # === Step 2: Loop over samples ===
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # --- Images ---
            img_files = row["image_list"].split(",")
            img_embeddings = []
            for file in img_files[:MAX_IMAGES]:
                img_path = os.path.join(IMAGE_FOLDER, file.strip())
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        img_feat = clip_model.encode_image(img_tensor).cpu().numpy()[0]  # (512,)
                    img_embeddings.append(img_feat)

            if len(img_embeddings) == 0:
                print(f"[Skipping] Sample {i} has no valid images.")
                continue

            # Pad or truncate to MAX_IMAGES
            if len(img_embeddings) < MAX_IMAGES:
                pad = [np.zeros(512, dtype=np.float32)] * (MAX_IMAGES - len(img_embeddings))
                img_embeddings.extend(pad)
            elif len(img_embeddings) > MAX_IMAGES:
                img_embeddings = img_embeddings[:MAX_IMAGES]

            # Option to average or keep full image tensor
            if USE_IMAGE_MEAN:
                img_vector = np.mean(np.stack(img_embeddings), axis=0)  # (512,)
                X_img.append(img_vector)
            else:
                img_tensor = np.stack(img_embeddings)  # (MAX_IMAGES, 512)
                X_img.append(img_tensor)

            # --- Text ---
            desc = row["Description"]
            tokens = clip_tokenizer([desc]).to(device)
            with torch.no_grad():
                text_feat = clip_model.encode_text(tokens).cpu().numpy()[0]  # (512,)
            X_text.append(text_feat)

            # --- Structured ---
            struct_feat = row[structured_columns].values.astype(np.float32)
            X_struct.append(struct_feat)

            # --- Label ---
            y.append(row["label"])

        except Exception as e:
            print(f"[Error] Sample {i}: {e}")
            continue

    print(f"Successfully processed {len(X_img)} samples with {MAX_IMAGES} images per sample.")

    # === Step 3: Normalize structured features ===
    X_struct = np.array(X_struct)
    X_struct = np.nan_to_num(X_struct)
    X_struct = (X_struct - X_struct.mean(axis=0)) / (X_struct.std(axis=0) + 1e-8)
    y = np.array(y)

    # === Step 4: Train/Val/Test Split ===
    X_img_train, X_img_temp, X_text_train, X_text_temp, X_struct_train, X_struct_temp, y_train, y_temp = train_test_split(
        X_img, X_text, X_struct, y, test_size=0.3, stratify=y, random_state=42)
    X_img_val, X_img_test, X_text_val, X_text_test, X_struct_val, X_struct_test, y_val, y_test = train_test_split(
        X_img_temp, X_text_temp, X_struct_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # === Step 5: Save to cache ===
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


# === Entry Point ===
if __name__ == "__main__":
    (X_img_train, X_text_train, X_struct_train, y_train), \
    (X_img_val, X_text_val, X_struct_val, y_val), \
    (X_img_test, X_text_test, X_struct_test, y_test) = load_text_image_structured_data()

    print("\nData loaded successfully!")
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"Text shape: {X_text_train.shape}, Structured: {X_struct_train.shape}")
    print(f"Image[0] shape: {X_img_train[0].shape}")
    print(f"Unique labels: {np.unique(y_train)}")