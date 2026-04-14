"""
Agent 2 - Infilteration Class-Specific Threshold Tuning

The false alarm problem in Agent 2 reduces to one confusion:
Benign records being classified as Infilteration (2,650 of 2,653 false alarms).

Root cause: Random Forest predicts Infilteration even at low confidence
(e.g. 42% vs 38% for Benign) because argmax treats all predictions equally.

Fix: only classify a record as Infilteration if confidence exceeds a tuned threshold.
Below that threshold, fall back to Benign.
This threshold is found by scanning values on the training set and picking the one that
minimises false alarm rate while preserving overall F1.

Outputs:
    - Best threshold value
    - Comparison: Agent 2 alone vs Agent 2 with threshold
    - Updated test predictions saved for Agent 3 to use
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, accuracy_score,
                             classification_report)
from sklearn.model_selection import StratifiedKFold

# Configuration

DATA_PATH = "../data/processed/cleaned_data.csv"
TRAIN_IDX = "../data/processed/train_idx.npy"
TEST_IDX = "../data/processed/test_idx.npy"
MODEL_PATH = "../data/processed/agent2_Random_Forest.pkl"
ENCODER_PATH = "../data/processed/label_encoder.pkl"
OUTPUT_DIR = "../data/processed"
LABEL_COL = "Label"
SEED = 42

# Load everything

print("Loading data and model...")
df = pd.read_csv(DATA_PATH)
le = joblib.load(ENCODER_PATH)
model = joblib.load(MODEL_PATH)
train_idx = np.load(TRAIN_IDX)
test_idx = np.load(TEST_IDX)

class_names = list(le.classes_)
infilter_idx = class_names.index("Infilteration")
benign_idx = class_names.index("Benign")
prob_cols = [f"prob_{c}" for c in class_names]

X = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL]

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

print(f"Classes: {class_names}")
print(f"Infilteration index: {infilter_idx}")


# Helper - apply threshold to probability matrix

def apply_infilteration_threshold(probs, base_preds, threshold, infilter_idx, benign_idx):
    """
    For any record predicted as Infilteration with probability below
    the threshold, reclassify as Benign.

    probs:        (n_samples, n_classes) probability matrix
    base_preds:   original argmax predictions
    threshold:    minimum confidence required to keep Infilteration prediction
    """
    adjusted = base_preds.copy()
    infilter_mask = (base_preds == infilter_idx)
    low_confidence = probs[:, infilter_idx] < threshold

    # Where both conditions are true - predicted Infilteration but not confident enough - reclassify as Benign
    adjusted[infilter_mask & low_confidence] = benign_idx
    return adjusted


def compute_far(y_true, y_pred, benign_idx):
    """False Alarm Rate: Benign records misclassified as attack."""
    benign_mask = (y_true == benign_idx)
    fp = np.sum((y_pred != benign_idx) & benign_mask)
    tn = np.sum((y_pred == benign_idx) & benign_mask)
    return fp / (fp + tn) if (fp + tn) > 0 else 0


# =============================================================================
# STEP 1: TUNE THRESHOLD ON TRAINING DATA VIA CROSS-VALIDATION
# =============================================================================
# Use 5-fold CV so we tune on held-out training folds - not the test set.
# For each candidate threshold, measure FAR and macro F1 on validation folds.
# Pick the threshold that minimises FAR while keeping F1 drop under 1%.
# =============================================================================

print("\n" + "=" * 60)
print("STEP 1: Tuning threshold on training data (5-fold CV)")
print("=" * 60)

thresholds = np.linspace(0.30, 0.90, 61)  # try 0.30, 0.31, ... 0.90
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

X_train_arr = X_train.values
y_train_arr = y_train.values

# Store average FAR and F1 across folds for each threshold
results = {t: {"far": [], "f1": []} for t in thresholds}

for fold, (fit_idx, val_idx) in enumerate(skf.split(X_train_arr,
                                                    y_train_arr)):
    print(f"  Fold {fold + 1}/5...")
    model.fit(X_train_arr[fit_idx], y_train_arr[fit_idx])

    val_probs = model.predict_proba(X_train_arr[val_idx])
    val_preds = model.predict(X_train_arr[val_idx])
    val_true = y_train_arr[val_idx]

    for t in thresholds:
        adj_preds = apply_infilteration_threshold(
            val_probs, val_preds, t, infilter_idx, benign_idx)
        results[t]["far"].append(compute_far(val_true, adj_preds, benign_idx))
        results[t]["f1"].append(f1_score(val_true, adj_preds,
                                         average="macro", zero_division=0))

# Average across folds
baseline_far = np.mean(results[thresholds[0]]["far"])
summary = {t: {
    "far": np.mean(results[t]["far"]),
    "f1": np.mean(results[t]["f1"])
} for t in thresholds}

print(f"\nBaseline FAR (threshold=0.30): {baseline_far:.4f}")
print(f"\nThreshold scan results (showing key points):")
print(f"{'Threshold':>10}  {'FAR':>8}  {'Macro F1':>10}")
for t in thresholds[::5]:
    print(f"  {t:.2f}       {summary[t]['far']:.4f}    {summary[t]['f1']:.4f}")

# Find baseline F1 (at threshold=0.30, very permissive)
baseline_f1 = summary[thresholds[0]]["f1"]

# Select best threshold - minimises FAR with F1 drop under 1%
best_threshold = thresholds[0]
best_far = summary[thresholds[0]]["far"]

for t in thresholds:
    f1_drop = baseline_f1 - summary[t]["f1"]
    if summary[t]["far"] < best_far and f1_drop < 0.01:
        best_far = summary[t]["far"]
        best_threshold = t

print(f"\nBest threshold found: {best_threshold:.2f}")
print(f"  CV FAR at this threshold: {best_far:.4f}")
print(f"  CV F1 at this threshold:  {summary[best_threshold]['f1']:.4f}")

# =============================================================================
# STEP 2: PLOT THRESHOLD SCAN
# =============================================================================

fars = [summary[t]["far"] for t in thresholds]
f1s = [summary[t]["f1"] for t in thresholds]

fig, ax1 = plt.subplots(figsize=(9, 4))
ax2 = ax1.twinx()

ax1.plot(thresholds, fars, color='tomato', linewidth=2, label='FAR')
ax2.plot(thresholds, f1s, color='steelblue', linewidth=2, label='Macro F1',
         linestyle='--')
ax1.axvline(best_threshold, color='gray', linestyle=':', linewidth=1.5,
            label=f'Best threshold ({best_threshold:.2f})')

ax1.set_xlabel("Infilteration Confidence Threshold")
ax1.set_ylabel("False Alarm Rate", color='tomato')
ax2.set_ylabel("Macro F1", color='steelblue')
ax1.set_title("Infilteration Threshold Tuning - FAR vs F1 Tradeoff")
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/infilteration_threshold_scan.png", dpi=150)
plt.close()
print(f"\nThreshold scan plot saved → {OUTPUT_DIR}/infilteration_threshold_scan.png")

# =============================================================================
# STEP 3: EVALUATE ON TEST SET WITH BEST THRESHOLD
# =============================================================================

print("\n" + "=" * 60)
print(f"STEP 2: Evaluating on test set (threshold={best_threshold:.2f})")
print("=" * 60)

# Retrain on full training set (CV used subsets)
print("Retraining on full training set...")
model.fit(X_train_arr, y_train_arr)

# Test predictions
test_probs = model.predict_proba(X_test.values)
test_preds = model.predict(X_test.values)
y_test_arr = y_test.values

# Apply threshold
adj_test_preds = apply_infilteration_threshold(test_probs, test_preds, best_threshold, infilter_idx, benign_idx)


# Evaluation
def evaluate(name, y_true, y_pred):
    y_true_names = le.inverse_transform(y_true)
    y_pred_names = le.inverse_transform(y_pred)
    print(f"\n--- {name} ---")
    print(classification_report(y_true_names, y_pred_names,
                                digits=4, zero_division=0))
    far = compute_far(y_true, y_pred, benign_idx)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    print(f"False Alarm Rate : {far:.4f}")
    print(f"Macro F1         : {f1:.4f}")
    print(f"Accuracy         : {acc:.4f}")
    return {"Accuracy": round(acc, 4),
            "Macro F1": round(f1, 4),
            "False Alarm Rate": round(far, 4)}


results_base = evaluate("Agent 2 - baseline (no threshold)", y_test_arr, test_preds)
results_adj = evaluate(f"Agent 2 - Infilteration threshold ({best_threshold:.2f})", y_test_arr, adj_test_preds)

# Summary
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
import pandas as pd

comparison = pd.DataFrame({
    "Agent 2 baseline": results_base,
    f"Agent 2 threshold={best_threshold:.2f}": results_adj
}).T
print(comparison.to_string())

far_change = results_adj["False Alarm Rate"] - results_base["False Alarm Rate"]
f1_change = results_adj["Macro F1"] - results_base["Macro F1"]
print(f"\nFAR change: {far_change:+.4f}  "
      f"({'IMPROVED ↓' if far_change < 0 else 'WORSENED ↑'})")
print(f"F1 change:  {f1_change:+.4f}")

# =============================================================================
# STEP 4: SAVE UPDATED PREDICTIONS FOR AGENT 3
# =============================================================================
# Save adjusted predictions so Agent 3 Bayesian fusion can use the threshold-corrected labels as a stronger prior.

prob_df = pd.DataFrame(test_probs, columns=prob_cols)
out_df = pd.DataFrame({
    "true_label": y_test_arr,
    "predicted": adj_test_preds,  # threshold-adjusted
    "predicted_raw": test_preds,  # original argmax
    **{c: prob_df[c] for c in prob_cols}
})
out_path = f"{OUTPUT_DIR}/agent2_Random_Forest_threshold_predictions.csv"
out_df.to_csv(out_path, index=False)
print(f"\nThreshold-adjusted predictions saved → {out_path}")
print(f"Best threshold: {best_threshold:.2f} - "
      f"use this value in production deployment")
