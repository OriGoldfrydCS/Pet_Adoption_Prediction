# This script splits the original dataset into train(70%), validation(15%), and test(15%)
import pandas as pd
import os
import chardet
from sklearn.model_selection import train_test_split

# --- Paths ---
INPUT_PATH = "dataset/data_processed/data/SD_TD_dataset.csv"
OUTPUT_DIR = "dataset/splits"

# --- Detect file encoding using chardet (to avoid encoding errors) ---
with open(INPUT_PATH, "rb") as f:
    result = chardet.detect(f.read(100000))
    encoding = result["encoding"]
    print(f"Detected encoding: {encoding}")

# --- Load dataset with detected encoding ---
df = pd.read_csv(INPUT_PATH, encoding=encoding)

# --- Drop rows without a label, if any ---
df = df.dropna(subset=["AdoptionSpeed"])

# --- First split: 70% train, 30% temp (for val + test) ---
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["AdoptionSpeed"],
    random_state=42
)

# --- Second split: 50% val, 50% test (i.e., each 15% of total) ---
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["AdoptionSpeed"],
    random_state=42
)

# --- Create output directory and save the splits ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
train_df.to_csv(os.path.join(OUTPUT_DIR, "train_SD_TD_dataset.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val_SD_TD_dataset.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test_SD_TD_dataset.csv"), index=False)

# --- Print summary ---
print("Splits created successfully:")
print("Train:", len(train_df), "| Val:", len(val_df), "| Test:", len(test_df))
print("Saved to:", OUTPUT_DIR)
