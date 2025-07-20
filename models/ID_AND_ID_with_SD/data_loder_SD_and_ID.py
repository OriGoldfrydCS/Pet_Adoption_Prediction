# This script loads structured features and corresponding images(mean images), processes them, normalizes the data, encodes labels, and returns train(70%)/val(15%)/test(15%) splits.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PIL import Image
import os

def load_image_and_structured_data():
    """
    Loads the CSV with structured features and image references,
    processes both modalities (structured + image), applies normalization, 
    and returns a dictionary with train/val/test splits for each input and the labels.
    """

    print("Loading structured data from CSV...")
    df = pd.read_csv("dataset/data_processed/SD_TD_ID_dataset.csv")
    print(f"Loaded {len(df)} samples.")

    # --- Extract structured features ---
    drop_columns = ["AdoptionSpeed", "Name", "Description", "PetID"]
    structured_cols = [
        col for col in df.columns
        if col not in drop_columns and df[col].dtype in [np.float32, np.float64, np.int64, np.int32]
    ]
    X_struct = df[structured_cols].values.astype(np.float32)
    X_struct = np.nan_to_num(X_struct, nan=0.0)

    # Normalize structured features
    scaler = StandardScaler()
    X_struct = scaler.fit_transform(X_struct)

    print(f"Structured feature shape: {X_struct.shape}")
    print(f"Used structured columns: {structured_cols}")
    print(f"NaNs in structured: {np.isnan(X_struct).sum()}, Infs: {np.isinf(X_struct).sum()}")
    print(f"Structured value range: {np.min(X_struct):.4f} to {np.max(X_struct):.4f}")

    # --- Process labels ---
    labels = df["AdoptionSpeed"].values
    labels = np.array([0 if x in [0, 1] else x for x in labels])  # Combine classes 0 and 1
    labels = LabelEncoder().fit_transform(labels)
    y = labels

    print(f"Label shape: {y.shape} | Unique classes: {np.unique(y)}")
    print("Final label mapping:", sorted(set(zip(df["AdoptionSpeed"].values, labels))))

    # --- Load and normalize images ---
    def load_images(row):
        """
        Loads one or more image files listed in the 'image_list' column,
        resizes them to 128x128, converts to RGB, and returns their mean image.
        If no image exists or loading fails, returns a zero-filled image.
        """
        if pd.isna(row["image_list"]):
            return np.zeros((128, 128, 3), dtype=np.float32)

        image_names = str(row["image_list"]).split(",")
        images = []

        for name in image_names:
            path = os.path.join("dataset", "original_datasets", "train_images", name.strip())
            try:
                img = Image.open(path).convert("RGB").resize((128, 128))
                img = np.array(img).astype(np.float32)
                images.append(img)
            except Exception as e:
                print(f"Error loading {path}: {e}")

        # Return mean image if any valid images loaded, else zero image
        return np.mean(images, axis=0).astype(np.float32) if images else np.zeros((128, 128, 3), dtype=np.float32)

    print("Loading and preprocessing images...")
    X_img = np.stack(df.apply(load_images, axis=1))
    print(f"Image tensor shape: {X_img.shape}")
    print(f"NaNs in images: {np.isnan(X_img).sum()}, Infs: {np.isinf(X_img).sum()}")
    print(f"Image value range: {np.min(X_img):.4f} to {np.max(X_img):.4f}")

    # --- Split into train/val/test ---
    print("Splitting data into train/val/test sets...")
    X_img_train, X_img_temp, X_struct_train, X_struct_temp, y_train, y_temp = train_test_split(
        X_img, X_struct, y, test_size=0.3, stratify=y, random_state=42
    )
    X_img_val, X_img_test, X_struct_val, X_struct_test, y_val, y_test = train_test_split(
        X_img_temp, X_struct_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print("Split sizes:")
    print(f"  Train: {X_img_train.shape[0]} samples")
    print(f"  Val:   {X_img_val.shape[0]} samples")
    print(f"  Test:  {X_img_test.shape[0]} samples")

    return {
        "X_img": (X_img_train, X_img_val, X_img_test),
        "X_struct": (X_struct_train, X_struct_val, X_struct_test),
        "y": (y_train, y_val, y_test)
    }

# --- Run test/debug mode ---
if __name__ == "__main__":
    data = load_image_and_structured_data()
    X_img_train, X_img_val, X_img_test = data["X_img"]
    X_struct_train, X_struct_val, X_struct_test = data["X_struct"]
    y_train, y_val, y_test = data["y"]

    print("\nShapes:")
    print(f"X_img_train: {X_img_train.shape}")
    print(f"X_struct_train: {X_struct_train.shape}")
    print(f"y_train: {y_train.shape}")

    print(f"\nTrain set: {len(X_img_train)} images | Class dist: {np.bincount(y_train)}")
    print(f"Val set:   {len(X_img_val)} images | Class dist: {np.bincount(y_val)}")
    print(f"Test set:  {len(X_img_test)} images | Class dist: {np.bincount(y_test)}")
