"""
Preprocessing Script — MA-NIDS
Loads raw CIC-IDS2018 CSVs, cleans them, balances classes,
encodes labels, and saves a single clean CSV + label encoder.
All agents load from this output — do not preprocess again.
"""

import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_DIR  = "../data/raw/ids-intrusion-csv"   # folder with raw CSVs
OUTPUT_DIR = "../data/processed"
SAMPLES    = 40_000    # max rows per class
MIN_SAMPLES = 10_000   # drop any class with fewer than this many rows
LABEL_COL  = "Label"
SEED       = 42
exclude_file_name = "02-20"

# Classes with too few samples to train reliably — remove them
DROP_CLASSES = [
    'Brute Force -Web',
    'Brute Force -XSS',
    'DDOS attack-LOIC-UDP',
    'SQL Injection',
    'DoS attacks-Slowloris',  # too few samples — only ~10k
    'Label'                   # artifact from bad concatenation
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Load one CSV file ──────────────────────────────────────────────────

def load_file(path):
    df = pd.read_csv(path, low_memory=False)

    # Column names sometimes have leading/trailing spaces e.g. " Label"
    # .str.strip() removes that so we can reliably find "Label" later
    df.columns = df.columns.str.strip()

    # These columns are identifiers, not traffic features
    # Including them would let the model cheat by memorising IDs instead of learning patterns
    drop_cols = ["Timestamp", "Dst IP", "Src IP", "Flow ID"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df

# ── Step 2: Clean one dataframe ───────────────────────────────────────────────

def clean(df):
    # Remove rows where the label is missing — we can't learn from unlabelled data
    df = df.dropna(subset=[LABEL_COL])

    # Strip whitespace from label values e.g. " Benign" → "Benign"
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()

    # Convert all feature columns to numbers
    # errors="coerce" means: if a value can't be converted, replace it with NaN
    # This handles things like "N/A" or stray text hiding in numeric columns
    feat_cols = [c for c in df.columns if c != LABEL_COL]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")

    # Some features produce division-by-zero during extraction → infinity
    # Models cannot handle infinity, so replace with NaN then drop
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df

# ── Step 3: Sample up to N rows per class ─────────────────────────────────────

def sample_per_class(df, n, seed):
    parts = []
    for _, group in df.groupby(LABEL_COL):
        # If a class has fewer than n rows, take all of them
        # If it has more, randomly sample n rows
        parts.append(group.sample(n=min(len(group), n), random_state=seed))
    return pd.concat(parts, ignore_index=True)

# ── Main Pipeline ─────────────────────────────────────────────────────────────

# Find all CSV files, skip 20-02 (different shape)
csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "**", "*.csv"), recursive=True))
csv_files = [f for f in csv_files if exclude_file_name not in os.path.basename(f)]
print(f"Found {len(csv_files)} files")

# Process one file at a time to avoid running out of memory
# After sampling each file, delete it from RAM before loading the next
all_samples = []
for path in csv_files:
    print(f"Processing: {os.path.basename(path)}")
    df = load_file(path)

    if LABEL_COL not in df.columns:
        print(f"  Skipping — no Label column found")
        continue

    df = clean(df)
    df = sample_per_class(df, SAMPLES, SEED)
    print(f"  Rows after sampling: {len(df)}")

    all_samples.append(df)
    del df    # free RAM before loading next file

# Combine all sampled files into one dataframe
combined = pd.concat(all_samples, ignore_index=True)
del all_samples

# Remove classes that were flagged as too rare to train on
combined = combined[~combined[LABEL_COL].isin(DROP_CLASSES)]

# Also drop any class that still has fewer than MIN_SAMPLES rows
# This catches any unexpected rare classes not in the DROP_CLASSES list
class_counts = combined[LABEL_COL].value_counts()
valid_classes = class_counts[class_counts >= MIN_SAMPLES].index
removed = class_counts[class_counts < MIN_SAMPLES].index.tolist()
if removed:
    print(f"Dropping classes with fewer than {MIN_SAMPLES} samples: {removed}")
combined = combined[combined[LABEL_COL].isin(valid_classes)]

# Re-sample to enforce balance across the combined dataset
# Per-file sampling wasn't enough because Benign appears in every file
# This final pass caps every class at SAMPLES rows
combined = sample_per_class(combined, SAMPLES, SEED)
print(f"\nAfter final balancing:")
print(combined[LABEL_COL].value_counts().sort_values())

# Shuffle the entire dataset
# Without this, all rows of class 0 come first, then class 1, etc.
# Models trained on ordered data learn order, not patterns
combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"\nCombined shape: {combined.shape}")

# ── Step 4: Encode labels ──────────────────────────────────────────────────────

# Models need numbers, not strings
# LabelEncoder maps each unique class name to an integer alphabetically
# e.g. Benign→0, BruteForce→1, DoS→2 ...
le = LabelEncoder()
combined[LABEL_COL] = le.fit_transform(combined[LABEL_COL])

print("\nLabel encoding map:")
for i, cls in enumerate(le.classes_):
    print(f"  {i} → {cls}")

# ── Step 5: Save outputs ───────────────────────────────────────────────────────

combined.to_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), index=False)

# Save the encoder so Agent 2 and Agent 3 can map numbers back to class names
joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

print(f"\nSaved cleaned_data.csv and label_encoder.pkl to {OUTPUT_DIR}")
