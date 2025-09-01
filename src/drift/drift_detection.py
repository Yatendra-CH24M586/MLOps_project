# src-drift-files/drift_detection.py

import os
import pandas as pd
from scipy.stats import ks_2samp
import subprocess

# Paths
TRAIN_DATA = "data/raw/train.csv"        # original training data
NEW_DATA = "data/raw/new_batch.csv"      # simulated new data batch

# Load data
train_df = pd.read_csv(TRAIN_DATA)
new_df = pd.read_csv(NEW_DATA)

# Features to check
columns_to_check = ["Age", "Fare"]

# Threshold for drift detection
DRIFT_THRESHOLD = 0.2
drift_detected = False

# Run KS test for each column
for col in columns_to_check:
    score = ks_2samp(train_df[col].dropna(), new_df[col].dropna()).pvalue
    if score < DRIFT_THRESHOLD:
        print(f"{col}: score={score:.3f} → ⚠️ Drift detected")
        drift_detected = True
    else:
        print(f"{col}: score={score:.3f} → ✅ OK")

# If drift detected → trigger retrain
if drift_detected:
    print("⚡ Drift detected! Triggering retraining pipeline...")
    subprocess.run(["python", "src/drift/retrain.py"], check=True)
else:
    print("✅ No significant drift. No retraining required.")
