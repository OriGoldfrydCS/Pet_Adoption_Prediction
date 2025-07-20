# For all the 58K images from the PetFinder dataset, this script loads images, resizes them to 128x128, normalizes pixel values to [0, 1], and saves them as .npy files.
import pandas as pd
import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

# --- Paths ---
IMAGE_DIR = "dataset/original_datasets/data_images"
LABELS_CSV = "dataset/original_datasets/data/original_data.csv"
OUTPUT_DIR = "dataset/data_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load PetID → AdoptionSpeed mapping from CSV ---
df = pd.read_csv(LABELS_CSV)
petid_to_label = dict(zip(df["PetID"], df["AdoptionSpeed"]))

# --- Prepare containers for processed images and their labels ---
images = []
labels = []

# --- List all image files in the folder ---
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]

# --- Go over every image file and load it if label exists ---
for img_file in tqdm(image_files, desc="Loading ALL images"):
    pet_id = img_file.split("-")[0]
    label = petid_to_label.get(pet_id)

    if label is not None:
        img_path = os.path.join(IMAGE_DIR, img_file)
        try:
            img = load_img(img_path, target_size=(128, 128))  # resize to fixed size
            img = img_to_array(img) / 255.0  # normalize to [0, 1]
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

# --- Convert lists to NumPy arrays ---
images = np.array(images)
labels = np.array(labels)

# --- Save as .npy and .pkl ---
np.save(os.path.join(OUTPUT_DIR, "images.npy"), images)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)
with open(os.path.join(OUTPUT_DIR, "image_data.pkl"), "wb") as f:
    pickle.dump({"images": images, "labels": labels}, f)

print(f"\nSaved {len(images)} images to '{OUTPUT_DIR}'")
