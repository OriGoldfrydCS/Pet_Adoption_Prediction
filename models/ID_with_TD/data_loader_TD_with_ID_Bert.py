# This script processes structured data with text and image(mean images) inputs. It encodes text using BERT, averages multiple images per sample, and returns train/val/test splits with cached embeddings.

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

# === Constants ===
CSV_PATH = "dataset/data_processed/SD_TD_ID_dataset.csv"
IMAGE_FOLDER = "dataset/original_datasets/train_images"
IMAGE_SIZE = (128, 128)
MAX_SEQ_LEN = 128
BERT_MODEL_NAME = "bert-base-uncased"
ENCODED_SAVE_DIR = "encoded_data"

# === Set device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load BERT model and tokenizer ===
print("Loading BERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
model = BertModel.from_pretrained(BERT_MODEL_NAME).to(device)
model.eval()


def encode_bert(sentences, mode="all", max_length=MAX_SEQ_LEN, batch_size=32, cache_file=None):
    """
    Encode a list of sentences using BERT.

    Args:
        sentences (list[str]): List of sentences to encode.
        mode (str): Encoding mode - 'cls', 'mean', or 'all'.
        max_length (int): Max token length for BERT.
        batch_size (int): Number of samples per batch.
        cache_file (str): Optional path to load/save cached numpy output.

    Returns:
        np.ndarray: Encoded BERT vectors (shape depends on mode).
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading BERT embeddings from cache: {cache_file}")
        return np.load(cache_file, allow_pickle=True)

    all_outputs = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding with BERT"):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state

        if mode == "all":
            # Pad to fixed sequence length
            padded = torch.zeros((last_hidden.size(0), max_length, last_hidden.size(2)), device=last_hidden.device)
            seq_len = last_hidden.size(1)
            padded[:, :seq_len, :] = last_hidden[:, :max_length, :]
            result = padded

        elif mode == "mean":
            result = last_hidden.mean(dim=1)
        elif mode == "cls":
            result = last_hidden[:, 0, :]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        all_outputs.append(result.cpu().numpy())

    final_output = (
        np.concatenate(all_outputs, axis=0) if mode == "all" else np.vstack(all_outputs)
    )

    if cache_file:
        if not os.path.exists(ENCODED_SAVE_DIR):
            os.makedirs(ENCODED_SAVE_DIR)
        np.save(cache_file, final_output)
        print(f"Saved BERT embeddings to {cache_file}")

    return final_output


def load_text_image_data(bert_mode="all"):
    """
    Load and preprocess both text (BERT) and image (averaged) data.

    Args:
        bert_mode (str): 'cls', 'mean', or 'all' - encoding strategy for BERT.

    Returns:
        tuple: Splits for train, validation, and test datasets.
    """
    print(f"Loading CSV and using BERT mode: '{bert_mode}'")
    df = pd.read_csv(CSV_PATH)

    # Clean and filter valid rows
    df = df.dropna(subset=["Description", "image_list", "AdoptionSpeed"]).reset_index(drop=True)
    df = df[df["Description"].str.strip() != ""]
    df = df[df["image_list"].str.strip() != ""]

    # Map target variable to 4-class label
    print("Mapping labels...")
    df["label"] = df["AdoptionSpeed"].map(lambda x: 0 if x in [0, 1] else (1 if x == 2 else (2 if x == 3 else 3)))

    # Cache paths
    text_cache = os.path.join(ENCODED_SAVE_DIR, f"X_text_all_{bert_mode}.npy")
    img_cache = os.path.join(ENCODED_SAVE_DIR, f"X_img_{bert_mode}.npy")
    y_cache = os.path.join(ENCODED_SAVE_DIR, f"y_{bert_mode}.npy")

    if os.path.exists(text_cache) and os.path.exists(img_cache) and os.path.exists(y_cache):
        print(f"Loading full dataset from cache (mode={bert_mode})...")
        X_text_all = np.load(text_cache, allow_pickle=True)
        X_img = np.load(img_cache)
        y = np.load(y_cache)
    else:
        # Encode text
        print("Encoding text with BERT...")
        descriptions = df["Description"].tolist()
        X_text_all = encode_bert(descriptions, mode=bert_mode, cache_file=text_cache)

        # Load and average images
        print("Loading and averaging images...")
        X_img, y = [], []

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            img_files = row["image_list"].split(",")
            imgs = []

            for file in img_files:
                img_path = os.path.join(IMAGE_FOLDER, file.strip())
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
                        imgs.append(np.array(img, dtype=np.float32) / 255.0)
                    except Exception as e:
                        print(f"Image load error: {img_path} — {e}")
                        continue

            if len(imgs) > 0:
                avg_img = np.mean(imgs, axis=0)
                X_img.append(avg_img)
                y.append(row["label"])

        # Sanity check
        assert len(X_img) == len(y) == len(X_text_all), "Mismatch in loaded sample sizes."

        X_img = np.array(X_img, dtype=np.float32)
        X_text_all = np.array(X_text_all, dtype=np.float32) if bert_mode != "all" else np.array(X_text_all)
        y = np.array(y)

        if not os.path.exists(ENCODED_SAVE_DIR):
            os.makedirs(ENCODED_SAVE_DIR)
        np.save(img_cache, X_img)
        np.save(y_cache, y)

    # Split into train/val/test (70/15/15 stratified)
    print("Splitting into train/val/test...")
    X_img_train, X_img_temp, X_text_train, X_text_temp, y_train, y_temp = train_test_split(
        X_img, X_text_all, y, test_size=0.3, stratify=y, random_state=42)

    X_img_val, X_img_test, X_text_val, X_text_test, y_val, y_test = train_test_split(
        X_img_temp, X_text_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    return (
        (X_img_train, X_text_train, y_train),
        (X_img_val, X_text_val, y_val),
        (X_img_test, X_text_test, y_test)
    )


if __name__ == "__main__":
    # Run the full pipeline and display basic stats
    mode = "all"  # change to "mean" or "cls" as needed
    (X_img_train, X_text_train, y_train), \
    (X_img_val, X_text_val, y_val), \
    (X_img_test, X_text_test, y_test) = load_text_image_data(bert_mode=mode)

    print("\nDone.")
    print(f"Train size: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"Image shape: {X_img_train.shape[1:]}")
    if mode == "all":
        print(f"Text shape: {X_text_train.shape}  (sequence of tokens)")
    else:
        print(f"Text shape: {X_text_train.shape}  (vector per sample)")
