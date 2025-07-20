# This script loads preprocessed image data and labels from .npy files, combines classes 0 and 1, encodes the labels, performs a stratified train(70%)/val(15%)/test(15%) split, and returns PyTorch DataLoaders along with raw image and label arrays for later use.

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Paths to image and label files ---
IMAGE_PATH = "dataset/data_processed/images.npy"
LABEL_PATH = "dataset/data_processed/labels.npy"

class ImageDataset(Dataset):
    """
    PyTorch Dataset for image classification.
    Uses index-based access to avoid unnecessary data duplication.
    """
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        image = self.images[i].astype(np.float32)
        image = torch.tensor(image).permute(2, 0, 1)  # Convert (H, W, C) → (C, H, W)
        label = torch.tensor(self.labels[i], dtype=torch.long)
        return image, label


def load_image_data(image_path=IMAGE_PATH, label_path=LABEL_PATH, random_state=4, batch_size=64):
    """
    Loads image data and labels, applies class re-grouping and label encoding,
    splits into train/val/test (stratified), and returns DataLoaders
    """
    print("Loading images and labels...")
    images = np.load(image_path, mmap_mode='r')  # Use memory-mapped mode to avoid RAM overload
    labels = np.load(label_path)

    print(f"Original label distribution: {np.unique(labels, return_counts=True)}")

    # Combine class 0+1 into a single class for binary classification baseline
    labels = np.array([0 if x in [0, 1] else x for x in labels])
    labels = LabelEncoder().fit_transform(labels)

    print(f"Labels after re-mapping + encoding: {np.unique(labels, return_counts=True)}")
    print(f"Total samples: {len(labels)} | Image shape: {images.shape[1:]}")

    # --- Create train/val/test splits (70/15/15) ---
    indices = np.arange(len(labels))
    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, test_size=0.3, stratify=labels, random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=random_state
    )

    # --- Wrap each split in a PyTorch Dataset ---
    train_dataset = ImageDataset(images, labels, train_idx)
    val_dataset = ImageDataset(images, labels, val_idx)
    test_dataset = ImageDataset(images, labels, test_idx)

    # --- Print class distribution ---
    print("Train label distribution:", np.bincount(labels[train_idx]))
    print("Val label distribution:", np.bincount(labels[val_idx]))
    print("Test label distribution:", np.bincount(labels[test_idx]))

    # --- Wrap datasets with DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nLoad complete:")
    print(f"Train set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")
    print(f"Test set: {len(test_dataset)} images")

    # --- Sanity check: show batch shape and values ---
    for xb, yb in train_loader:
        print(f"Sanity check batch: x.shape = {xb.shape}, y.shape = {yb.shape}")
        print(f"Value range: min={xb.min():.3f}, max={xb.max():.3f}")
        print(f"First labels: {yb[:8].tolist()}")
        break

    return train_loader, val_loader, test_loader, images, labels, val_idx, test_idx


# --- Run quick test if executed as main script ---
if __name__ == "__main__":
    train_loader, val_loader, test_loader, all_images, all_labels, val_idx, test_idx = load_image_data()
    
    # Optional: create full tensors for validation/testing (useful for inference or metrics later)
    X_val_tensor = torch.tensor(all_images[val_idx]).permute(0, 3, 1, 2).float()
    X_test_tensor = torch.tensor(all_images[test_idx]).permute(0, 3, 1, 2).float()
