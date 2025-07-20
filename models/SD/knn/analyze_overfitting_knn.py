"""
Script for analyzing overfitting in KNN by sweeping across different k values.

What it does:
- Iteratively trains KNN models with different values of k (1 to 150, step 5)
- For each run, reads saved metrics from the latest output folder
- Computes train-test accuracy gap and log loss gap
- Plots both gaps to visualize overfitting trends

Requirements:
- Each run saves metrics in `val_metrics.txt` with the following lines:
    0: Test Accuracy
    1: Test F1
    2: Test Log Loss
    3: Train Accuracy
    4: Train Log Loss
"""

import subprocess
import os
import time
import matplotlib.pyplot as plt
import re

# Output folder for saving the analysis plots
output_dir = os.path.join("performance_and_models", "knn", "analyze_overfitting_knn")
os.makedirs(output_dir, exist_ok=True)

# Range of K values to test
k_values = list(range(1, 151, 5))
accuracy_gaps = []
logloss_gaps = []

# Sweep over k
for k in k_values:
    print(f"Running k={k}")

    # Run training script with current k
    subprocess.run(["python", "models/SD/knn/main.py", str(k)])

    # Find the latest output folder (timestamped)
    base_dir = os.path.join("performance_and_models", "knn")
    run_folders = sorted(
        [d for d in os.listdir(base_dir) if re.match(r"\d{8}_\d{6}", d)],
        reverse=True
    )

    if not run_folders:
        print(f"No valid run folders found after k={k}. Skipping...")
        continue

    latest_run = run_folders[0]
    metrics_path = os.path.join(base_dir, latest_run, "val_metrics.txt")

    if not os.path.exists(metrics_path):
        print(f"val_metrics.txt not found for k={k}. Skipping...")
        continue

    try:
        # Read metrics and compute gap
        with open(metrics_path, "r") as f:
            lines = f.readlines()
            test_acc = float(lines[0].split(":")[1])
            train_acc = float(lines[3].split(":")[1])
            acc_gap = train_acc - test_acc
            accuracy_gaps.append(acc_gap)

            test_logloss = float(lines[2].split(":")[1])
            train_logloss = float(lines[4].split(":")[1])
            logloss_gap = test_logloss - train_logloss
            logloss_gaps.append(logloss_gap)

    except Exception as e:
        print(f"Error reading metrics for k={k}: {e}")
        continue

    # Optional: allow time for OS to settle before next run
    time.sleep(1)

# --- Plot Accuracy Gap ---
plt.figure(figsize=(10, 5))
plt.plot(k_values[:len(accuracy_gaps)], accuracy_gaps, marker='o')
plt.title("Train–Test Accuracy Gap vs K")
plt.xlabel("K (n_neighbors)")
plt.ylabel("Accuracy Gap (Train - Test)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "accuracy_gap_vs_k.png"))
plt.show()

# --- Plot Log Loss Gap ---
plt.figure(figsize=(10, 5))
plt.plot(k_values[:len(logloss_gaps)], logloss_gaps, marker='o', color='red')
plt.title("Test–Train Log Loss Gap vs K")
plt.xlabel("K (n_neighbors)")
plt.ylabel("Log Loss Gap (Test - Train)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "logloss_gap_vs_k.png"))
plt.show()
