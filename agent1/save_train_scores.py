"""
Agent 1 - Save Training Scores for Agent 3
Loads already-trained Agent 1 models and computes anomaly scores
on the full training set. Agent 3 needs these to estimate
per-class likelihood distributions for Bayesian fusion.

Run this AFTER agent1.py has completed successfully.
"""

import joblib
import numpy as np
import pandas as pd
import torch

from agent1 import Autoencoder, get_autoencoder_scores, get_isolation_forest_scores, normalize_scores, CONFIG

# Paths

DATA_PATH = "../data/processed/cleaned_data.csv"
TRAIN_IDX_PATH = "../data/processed/train_idx.npy"
SCALER_PATH = "../data/processed/agent1_outputs/agent1_scaler.pkl"
IF_PATH = "../data/processed/agent1_outputs/isolation_forest.pkl"
AE_PATH = "../data/processed/agent1_outputs/autoencoder.pt"
OUTPUT_PATH = "../data/processed/agent1_outputs/agent1_train_scores.csv"

LABEL_COL = "Label"
ENCODING_DIM = CONFIG["ae_encoding_dim"]

# Main

print("Loading data...")
df = pd.read_csv(DATA_PATH)
train_idx = np.load(TRAIN_IDX_PATH)

X = df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y_raw = df[LABEL_COL].values

# Apply same scaling + clipping as agent1.py
scaler = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X)
X_scaled = np.clip(X_scaled, -10, 10)

# Get full training set with all classes
X_train_all = X_scaled[train_idx]
y_train_multiclass = y_raw[train_idx]

print(f"Full training set: {X_train_all.shape[0]:,} records")
print(f"Classes present: {np.unique(y_train_multiclass)}")

# Load trained models
print("\nLoading trained models...")
iso_forest = joblib.load(IF_PATH)

n_features = X_train_all.shape[1]
ae_model = Autoencoder(n_features, ENCODING_DIM)
ae_model.load_state_dict(torch.load(AE_PATH, weights_only=True))
ae_model.eval()
print("Models loaded.")

# Compute scores on full training set
print("\nComputing training scores...")
ae_scores = get_autoencoder_scores(ae_model, X_train_all)
if_scores = get_isolation_forest_scores(iso_forest, X_train_all)

ae_norm = normalize_scores(ae_scores)
if_norm = normalize_scores(if_scores)
combined = (1.0 * ae_norm) + (0.0 * if_norm)

# Save with multiclass labels - Agent 3 fits one Gaussian per class
train_scores_df = pd.DataFrame({
    "ae_score_normalized": ae_norm,
    "if_score_normalized": if_norm,
    "agent1_combined_score": combined,
    "true_label_multiclass": y_train_multiclass
})
train_scores_df.to_csv(OUTPUT_PATH, index=False)
print(f"Training scores saved → {OUTPUT_PATH}")

# Quick sanity check - mean score per class
le = joblib.load("../data/processed/label_encoder.pkl")
train_scores_df["class_name"] = le.inverse_transform(y_train_multiclass)
print("\nMean combined score per class (training set):")
print(train_scores_df.groupby("class_name")["agent1_combined_score"]
      .mean().sort_values(ascending=False))
print("\nDone. Agent 3 can now load agent1_train_scores.csv.")
