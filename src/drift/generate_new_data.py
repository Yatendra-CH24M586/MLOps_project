# src-drift-files/data_generator.py

import pandas as pd
import numpy as np
import os

# Paths
RAW_PATH = "data/raw/train.csv"
NEW_BATCH_PATH = "data/raw/new_batch.csv"

# Load training data
df = pd.read_csv(RAW_PATH)

# Sample 200 random rows
new_batch = df.sample(200, replace=True).copy()

# Simulate drift: increase "Fare", shift "Age"
new_batch["Fare"] = new_batch["Fare"] * np.random.uniform(1.2, 1.5, size=len(new_batch))
new_batch["Age"] = new_batch["Age"] + np.random.randint(-5, 5, size=len(new_batch))

# Ensure no negative ages
new_batch["Age"] = new_batch["Age"].clip(lower=0)

# Save
os.makedirs(os.path.dirname(NEW_BATCH_PATH), exist_ok=True)
new_batch.to_csv(NEW_BATCH_PATH, index=False)

print(f"✅ New batch generated at {NEW_BATCH_PATH}")
