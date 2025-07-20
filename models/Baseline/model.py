# Baseline Model_0: Majority Class Predictor
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from datetime import datetime

# --- Paths ---
CSV_PATH = "dataset/data_processed/data/data_Ucb_dupQ_final.csv"
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
base_output_dir = os.path.join("performance_and_models", "model_0", run_id)
os.makedirs(base_output_dir, exist_ok=True)

# --- Load data ---
df = pd.read_csv(CSV_PATH)
y_true = df["AdoptionSpeed"]

# --- Predict the majority class ---
majority_class = y_true.value_counts().idxmax()
y_pred = [majority_class] * len(y_true)

# --- Calculate metrics ---
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

# --- For loss: convert to probabilities (one-hot like)
num_classes = y_true.nunique()
y_prob = np.zeros((len(y_true), num_classes))
y_prob[:, majority_class] = 1  # probability 1 for majority class
loss = log_loss(y_true, y_prob)

# --- Save results ---
metrics_path = os.path.join(base_output_dir, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write("Baseline (Majority Class Predictor)\n")
    f.write(f"Predicted class: {majority_class}\n\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")
    f.write(f"Loss:      {loss:.4f}\n")

print(f"Done. Results saved to: {metrics_path}")
