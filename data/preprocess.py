"""
Preprocessing Script - MA-NIDS
Loads raw CIC-IDS2018 CSVs, cleans them, balances classes,
encodes labels, and saves a single clean CSV + label encoder.
Also saves shared train/test indices so all agents evaluate
on identical records.
All agents load from this output - do not preprocess again.
"""

import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_DIR = "../data/raw/ids-intrusion-csv"
OUTPUT_DIR = "../data/processed"
SAMPLES = 40_000
MIN_SAMPLES = 10_000
LABEL_COL = "Label"
SEED = 42
exclude_file_name = "02-20"

DROP_CLASSES = [
    'Brute Force -Web',
    'Brute Force -XSS',
    'DDOS attack-LOIC-UDP',
    'SQL Injection',
    'DoS attacks-Slowloris',
    'Label'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Step 1: Load one CSV file ──────────────────────────────────────────────────

def load_file(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    drop_cols = ["Timestamp", "Dst IP", "Src IP", "Flow ID"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df


# ── Step 2: Clean one dataframe ───────────────────────────────────────────────

def clean(df):
    df = df.dropna(subset=[LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    feat_cols = [c for c in df.columns if c != LABEL_COL]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


# ── Step 3: Sample up to N rows per class ─────────────────────────────────────

def sample_per_class(df, n, seed):
    parts = []
    for _, group in df.groupby(LABEL_COL):
        parts.append(group.sample(n=min(len(group), n), random_state=seed))
    return pd.concat(parts, ignore_index=True)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "**", "*.csv"),
                             recursive=True))
csv_files = [f for f in csv_files if exclude_file_name not in os.path.basename(f)]
print(f"Found {len(csv_files)} files")

all_samples = []
for path in csv_files:
    print(f"Processing: {os.path.basename(path)}")
    df = load_file(path)

    if LABEL_COL not in df.columns:
        print(f"  Skipping - no Label column found")
        continue

    df = clean(df)
    df = sample_per_class(df, SAMPLES, SEED)
    print(f"  Rows after sampling: {len(df)}")

    all_samples.append(df)
    del df

combined = pd.concat(all_samples, ignore_index=True)
del all_samples

# Remove rare and problematic classes
combined = combined[~combined[LABEL_COL].isin(DROP_CLASSES)]

# Drop any class still below minimum sample threshold
class_counts = combined[LABEL_COL].value_counts()
valid_classes = class_counts[class_counts >= MIN_SAMPLES].index
removed = class_counts[class_counts < MIN_SAMPLES].index.tolist()
if removed:
    print(f"Dropping classes with fewer than {MIN_SAMPLES} samples: {removed}")
combined = combined[combined[LABEL_COL].isin(valid_classes)]

# Re-sample globally to enforce balance across combined dataset
combined = sample_per_class(combined, SAMPLES, SEED)
print(f"\nAfter final balancing:")
print(combined[LABEL_COL].value_counts().sort_values())

# Shuffle entire dataset
combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"\nCombined shape: {combined.shape}")

# ── Step 4: Encode labels ──────────────────────────────────────────────────────

le = LabelEncoder()
combined[LABEL_COL] = le.fit_transform(combined[LABEL_COL])

print("\nLabel encoding map:")
for i, cls in enumerate(le.classes_):
    print(f"  {i} → {cls}")

# ── Step 5: Save outputs ───────────────────────────────────────────────────────

combined.to_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), index=False)
joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

# ── Step 6: Save shared train/test indices ────────────────────────────────────
# All agents load these indices to evaluate on identical records
# This ensures Agent 3 can join Agent 1 and Agent 2 outputs row by row

indices = np.arange(len(combined))
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=SEED,
    stratify=combined[LABEL_COL]
)

np.save(os.path.join(OUTPUT_DIR, "train_idx.npy"), train_idx)
np.save(os.path.join(OUTPUT_DIR, "test_idx.npy"), test_idx)

print(f"\nSaved cleaned_data.csv and label_encoder.pkl to {OUTPUT_DIR}")
print(f"Train indices saved: {len(train_idx):,} records")
print(f"Test indices saved:  {len(test_idx):,} records")
