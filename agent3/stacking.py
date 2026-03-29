"""
Agent 3 - Stacking (Meta-Learning Fusion)
Multi-Agent Network Intrusion Detection System (MA-NIDS)
CS 5100: Foundations of Artificial Intelligence

Alternative to Bayesian fusion. Trains a Logistic Regression meta-model
that learns how to combine Agent 1 and Agent 2 outputs from data,
rather than using a fixed mathematical formula.

How Stacking Works:
    1. Use 5-fold cross-validation on training data to generate
       Agent 2 probability predictions without data leakage
    2. Combine these with Agent 1 training scores to build meta-features
    3. Train Logistic Regression on meta-features
    4. At test time: combine Agent 1 + Agent 2 test outputs → meta-model → final prediction

Advantage over Bayesian fusion:
    - Learns combination rule from data, no Gaussian assumptions
    - Handles correlated agents naturally
    - Better at resolving ambiguous cases like Benign vs Infilteration

Outputs:
    - Per-record predicted class and threat score
    - Comparison: Agent 2 alone vs Bayesian fusion vs Stacking
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_PATH = "../data/processed/cleaned_data.csv"
TRAIN_IDX_PATH = "../data/processed/train_idx.npy"
TEST_IDX_PATH = "../data/processed/test_idx.npy"
ENCODER_PATH = "../data/processed/label_encoder.pkl"
AGENT1_TRAIN_SCORES = "../data/processed/agent1_outputs/agent1_train_scores.csv"
AGENT1_TEST_SCORES = "../data/processed/agent1_outputs/agent1_scores.csv"
AGENT2_TEST_PREDS = "../data/processed/agent2_Random_Forest_test_predictions.csv"
BAYESIAN_RESULTS = "../data/processed/agent3_outputs/agent3_comparison.csv"
OUTPUT_DIR = "../data/processed/agent3_outputs"
LABEL_COL = "Label"
N_FOLDS = 5
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Everything ───────────────────────────────────────────────────────────

warnings.filterwarnings("ignore", category=RuntimeWarning)

print("=" * 60)
print("STEP 1: Loading data and Agent outputs")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
le = joblib.load(ENCODER_PATH)
train_idx = np.load(TRAIN_IDX_PATH)
test_idx = np.load(TEST_IDX_PATH)

class_names = list(le.classes_)
n_classes = len(class_names)
prob_cols = [f"prob_{c}" for c in class_names]

# Features and labels
X = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL]

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

# Agent 1 scores
a1_train = pd.read_csv(AGENT1_TRAIN_SCORES)
a1_test = pd.read_csv(AGENT1_TEST_SCORES)

# Agent 2 test predictions - already saved from train.py
a2_test = pd.read_csv(AGENT2_TEST_PREDS)

print(f"Training records : {len(X_train):,}")
print(f"Test records     : {len(X_test):,}")
print(f"Classes          : {class_names}")

# =============================================================================
# STEP 2: BUILD META-TRAINING SET VIA CROSS-VALIDATION
# =============================================================================
# Agent 2 predicts on held-out folds so meta-model never sees predictions
# made on data Agent 2 trained on. This prevents data leakage.
#
# For each fold:
#   - Train RF on 4/5 of training data
#   - Predict probabilities on the held-out 1/5
#   - Combine with Agent 1 scores for those same records
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Building meta-training set via 5-fold cross-validation")
print("=" * 60)

# Agent 2 base model - same hyperparameters as train.py
base_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=SEED
)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Placeholder arrays for meta-features
# Shape: (n_train, n_classes + 1) - one col per class prob + agent1 score
meta_train_probs = np.zeros((len(X_train), n_classes))
meta_train_agent1 = np.zeros(len(X_train))

X_train_arr = X_train.values
y_train_arr = y_train.values

print(f"Running {N_FOLDS}-fold cross-validation...")

for fold, (fit_idx, val_idx) in enumerate(skf.split(X_train_arr, y_train_arr)):
    print(f"  Fold {fold + 1}/{N_FOLDS} - "
          f"train: {len(fit_idx):,}  val: {len(val_idx):,}")

    # Train Agent 2 on 4 folds
    base_model.fit(X_train_arr[fit_idx], y_train_arr[fit_idx])

    # Predict probabilities on held-out fold
    # These are honest predictions - Agent 2 never saw these records
    meta_train_probs[val_idx] = base_model.predict_proba(X_train_arr[val_idx])

    # Agent 1 score for these same held-out records
    meta_train_agent1[val_idx] = \
        a1_train["agent1_combined_score"].values[val_idx]

print("Cross-validation complete.")

# Build final meta-feature matrix for training
# Columns: [prob_class0, prob_class1, ..., agent1_score]
meta_X_train = np.column_stack([meta_train_probs, meta_train_agent1])
meta_y_train = y_train_arr

print(f"Meta-training set shape: {meta_X_train.shape}")

# =============================================================================
# STEP 3: TRAIN META-MODEL (LOGISTIC REGRESSION)
# =============================================================================
# Logistic Regression is the right choice here because:
#   - Meta-features are already rich probability signals - no need for
#     a complex model
#   - Interpretable - coefficients show how much each agent is trusted
#   - Doesn't overfit on the small meta-feature space (n_classes + 1 columns)
#   - Much faster than a second Random Forest
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Training Logistic Regression meta-model")
print("=" * 60)

meta_model = LogisticRegression(
    max_iter=1000,  # ensure convergence
    random_state=SEED,
    n_jobs=-1,
    C=1.0  # regularisation strength - prevents overfitting
)

meta_model.fit(meta_X_train, meta_y_train)
print("Meta-model trained.")

# Save meta-model
joblib.dump(meta_model, os.path.join(OUTPUT_DIR, "agent3_stacking_meta_model.pkl"))
print(f"Meta-model saved → {OUTPUT_DIR}/agent3_stacking_meta_model.pkl")

# Show feature importance - which agent does the meta-model trust more?
# Coefficients shape: (n_classes, n_features)
# Last column = agent1 coefficient per class
agent1_coefs = meta_model.coef_[:, -1]
print(f"\nAgent 1 score coefficient per class (higher = trusted more):")
for cls, coef in zip(class_names, agent1_coefs):
    print(f"  {cls:35s} : {coef:+.4f}")

# =============================================================================
# STEP 4: BUILD META-TEST SET AND PREDICT
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Predicting on test set")
print("=" * 60)

# Agent 2 test probabilities - already saved from train.py
# No cross-validation needed here - we use the full Agent 2 model's outputs
agent2_test_probs = a2_test[prob_cols].values

# Agent 1 test scores
agent1_test_scores = a1_test["agent1_combined_score"].values

# Build meta-test features
meta_X_test = np.column_stack([agent2_test_probs, agent1_test_scores])

# Predict with meta-model
stacking_preds = meta_model.predict(meta_X_test)
stacking_probs = meta_model.predict_proba(meta_X_test)

# Threat score = 1 - probability of Benign
benign_idx = class_names.index("Benign")
threat_scores = 1 - stacking_probs[:, benign_idx]

print(f"Predictions complete for {len(stacking_preds):,} test records.")

# =============================================================================
# STEP 5: EVALUATE AND COMPARE ALL THREE APPROACHES
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Evaluation - Agent 2 vs Bayesian vs Stacking")
print("=" * 60)


def compute_far(y_true, y_pred):
    """False Alarm Rate: Benign records misclassified as attack."""
    bidx = class_names.index("Benign")
    benign_mask = (y_true == bidx)
    fp = np.sum((y_pred != bidx) & benign_mask)
    tn = np.sum((y_pred == bidx) & benign_mask)
    return fp / (fp + tn) if (fp + tn) > 0 else 0


def evaluate(name, y_true, y_pred):
    y_true_names = le.inverse_transform(y_true)
    y_pred_names = le.inverse_transform(y_pred)

    print(f"\n--- {name} ---")
    print(classification_report(y_true_names, y_pred_names,
                                digits=4, zero_division=0))

    far = compute_far(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    print(f"False Alarm Rate : {far:.4f}")
    print(f"Macro F1         : {f1:.4f}")
    print(f"Accuracy         : {acc:.4f}")

    return {"Accuracy": round(acc, 4),
            "Macro F1": round(f1, 4),
            "False Alarm Rate": round(far, 4)}


true_labels = y_test.values
a2_preds = a2_test["predicted"].values

results_a2 = evaluate("Agent 2 - Random Forest alone",
                      true_labels, a2_preds)
results_stk = evaluate("Agent 3 - Stacking (LR meta-model)",
                       true_labels, stacking_preds)

# Load Bayesian results for three-way comparison
try:
    bayes_df = pd.read_csv(BAYESIAN_RESULTS, index_col=0)
    results_bay = bayes_df.loc["Agent 3 (Bayesian)"].to_dict()
    print(f"\nBayesian results loaded from {BAYESIAN_RESULTS}")
except Exception:
    results_bay = {"Accuracy": "N/A", "Macro F1": "N/A",
                   "False Alarm Rate": "N/A"}
    print("Bayesian results not found - run fusion.py first")

# Three-way comparison table
print("\n" + "=" * 60)
print("THREE-WAY COMPARISON SUMMARY")
print("=" * 60)
comparison = pd.DataFrame({
    "Agent 2 alone": results_a2,
    "Agent 3 Bayesian": results_bay,
    "Agent 3 Stacking": results_stk,
}).T
print(comparison.to_string())

# Key changes
far_change = results_stk["False Alarm Rate"] - results_a2["False Alarm Rate"]
f1_change = results_stk["Macro F1"] - results_a2["Macro F1"]
print(f"\nStacking vs Agent 2:")
print(f"  FAR change : {far_change:+.4f}  "
      f"({'IMPROVED ↓' if far_change < 0 else 'WORSENED ↑'})")
print(f"  F1 change  : {f1_change:+.4f}")


# =============================================================================
# STEP 6: CONFUSION MATRIX AND THREAT SCORE PLOT
# =============================================================================

def plot_cm(y_true, y_pred, title, fname):
    cm = confusion_matrix(le.inverse_transform(y_true),
                          le.inverse_transform(y_pred),
                          labels=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Greens')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")


print("\nSaving plots...")
plot_cm(true_labels, stacking_preds,
        "Agent 3 - Stacking (LR meta-model)",
        "cm_agent3_stacking.png")

# Threat score distribution
benign_mask = (true_labels == benign_idx)
attack_mask = ~benign_mask

plt.figure(figsize=(8, 4))
plt.hist(threat_scores[benign_mask], bins=50, alpha=0.6,
         color='steelblue', label='Benign', density=True)
plt.hist(threat_scores[attack_mask], bins=50, alpha=0.6,
         color='tomato', label='Attack', density=True)
plt.title("Agent 3 Stacking - Threat Score Distribution")
plt.xlabel("Threat Score (higher = more likely an attack)")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,
                         "threat_score_distribution_stacking.png"), dpi=150)
plt.close()
print("Saved: threat_score_distribution_stacking.png")

# =============================================================================
# STEP 7: SAVE RESULTS
# =============================================================================

results_df = pd.DataFrame({
    "true_label": true_labels,
    "true_class": le.inverse_transform(true_labels),
    "agent2_predicted": a2_preds,
    "stacking_predicted": stacking_preds,
    "threat_score": threat_scores,
    **{f"meta_prob_{c}": stacking_probs[:, i]
       for i, c in enumerate(class_names)}
})

results_path = os.path.join(OUTPUT_DIR, "agent3_stacking_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\nFull results saved → {results_path}")

comparison.to_csv(os.path.join(OUTPUT_DIR, "agent3_full_comparison.csv"))
print(f"Three-way comparison saved → {OUTPUT_DIR}/agent3_full_comparison.csv")

print("\n" + "=" * 60)
print("AGENT 3 STACKING COMPLETE")
print("=" * 60)
