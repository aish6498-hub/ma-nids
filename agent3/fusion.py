"""
Agent 3 - Bayesian Fusion

Combines Agent 1 (anomaly score) and Agent 2 (class probabilities)
using Bayes' Theorem to produce a calibrated posterior threat probability.

How Bayesian Fusion Works:
    posterior(class) ∝ prior(class) × likelihood(agent1_score | class)

    - prior(class)              = Agent 2's class probability
    - likelihood(score | class) = probability of this Agent 1 score given the record belongs to this class,
                                  estimated as a Gaussian fitted on training data per class
    - posterior(class)          = updated belief after combining both
    - threat_score              = 1 - posterior(Benign)

Known limitation:
    In practice, the improved autoencoder produces anomaly scores that are
    uniformly low across all classes (means between 0.0005 and 0.0065).
    All per-class Gaussians hit the minimum std floor of 0.05, making
    their likelihood values nearly identical for any given score.
    When likelihoods are identical across classes, the posterior cannot
    move away from Agent 2's prior - fusion is mathematically inert.
    See agent3/stacking.py for the approach that does produce improvement.

Outputs:
    - agent3_results.csv      : per-record predictions, threat scores, and full posterior distributions
    - agent3_comparison.csv   : Agent 2 alone vs Agent 3 metrics summary
    - cm_agent2.png           : Agent 2 confusion matrix
    - cm_agent3.png           : Agent 3 confusion matrix
    - threat_score_distribution.png : Benign vs attack threat score separation
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

# Configuration

AGENT1_TEST_SCORES = "../data/processed/agent1_outputs/agent1_scores.csv"
AGENT1_TRAIN_SCORES = "../data/processed/agent1_outputs/agent1_train_scores.csv"
AGENT2_PREDICTIONS = "../data/processed/agent2_Random_Forest_test_predictions.csv"
ENCODER_PATH = "../data/processed/label_encoder.pkl"
OUTPUT_DIR = "../data/processed/agent3_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

print("=" * 60)
print("STEP 1: Loading Agent 1 and Agent 2 outputs")
print("=" * 60)

le = joblib.load(ENCODER_PATH)
class_names = list(le.classes_)
n_classes = len(class_names)

# Agent 1 test scores - one row per test record
a1_test = pd.read_csv(AGENT1_TEST_SCORES)

# Agent 1 training scores - used to fit likelihood distributions
a1_train = pd.read_csv(AGENT1_TRAIN_SCORES)

# Agent 2 predictions and probabilities - one row per test record
a2_test = pd.read_csv(AGENT2_PREDICTIONS)

# Verify alignment - both files must have same number of rows
assert len(a1_test) == len(a2_test), \
    f"Row mismatch: Agent1={len(a1_test)}, Agent2={len(a2_test)}"

print(f"Test records     : {len(a1_test):,}")
print(f"Training records : {len(a1_train):,}")
print(f"Classes          : {class_names}")

# Agent 1 combined score for each test record
agent1_scores = a1_test["agent1_combined_score"].values

# Agent 2 probability columns - one per class
# Column names follow pattern: prob_ClassName
prob_cols = [f"prob_{c}" for c in class_names]
agent2_probs = a2_test[prob_cols].values  # shape (n_test, n_classes)

# True multiclass labels for evaluation
# Agent 1 scores.csv uses true_label_multiclass
# Agent 2 predictions.csv uses true_label
true_labels = a2_test["true_label"].values

# =============================================================================
# STEP 2: ESTIMATE LIKELIHOOD DISTRIBUTIONS
# =============================================================================
# For each class, fit a Gaussian (normal distribution) to the Agent 1 scores
# of all training records belonging to that class.
#
# This gives us P(agent1_score | class) - the likelihood.
# At inference: a new record with agent1_score=0.8 is very likely to be a class whose training records had high scores,
# and unlikely to be a class whose training records had low scores.
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Estimating per-class likelihood distributions")
print("=" * 60)

# Store Gaussian parameters (mean, std) for each class
likelihoods = {}

for class_idx, class_name in enumerate(class_names):
    # Get Agent 1 scores for all training records of this class
    class_mask = a1_train["true_label_multiclass"] == class_idx
    class_scores = a1_train.loc[class_mask, "agent1_combined_score"].values

    if len(class_scores) == 0:
        # Fallback if no training samples for this class
        likelihoods[class_idx] = {"mean": 0.5, "std": 0.1}
        print(f"  {class_name:35s} - NO TRAINING SAMPLES (using fallback)")
        continue

    mu = class_scores.mean()
    # Minimum std floor of 0.05 prevents likelihood collapse for classes where Agent 1 scores have very low variance
    # (e.g. FTP-BruteForce and SlowHTTPTest both had std ~0.0004 which caused their posteriors to collapse to zero)
    sigma = max(class_scores.std(), 0.05) + 1e-6

    likelihoods[class_idx] = {"mean": mu, "std": sigma}
    print(f"  {class_name:35s} - mean: {mu:.4f}  std: {sigma:.4f}  "
          f"n: {len(class_scores):,}")

# =============================================================================
# STEP 3: BAYESIAN FUSION
# =============================================================================
# For each test record:
#   1. Get prior from Agent 2: P(class) = agent2_prob[class]
#   2. Compute likelihood from Gaussian: P(score | class)
#   3. Posterior ∝ prior × likelihood
#   4. Normalise posteriors to sum to 1
#   5. predicted_class = argmax(posterior)
#   6. threat_score = 1 - posterior[Benign]
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Bayesian Fusion")
print("=" * 60)

benign_idx = class_names.index("Benign")
n_test = len(agent1_scores)

posteriors = np.zeros((n_test, n_classes))
predicted_class = np.zeros(n_test, dtype=int)
threat_scores = np.zeros(n_test)

print(f"Fusing {n_test:,} test records...")

for i in range(n_test):
    score = agent1_scores[i]
    priors = agent2_probs[i]  # Agent 2 probability vector

    for c in range(n_classes):
        mu = likelihoods[c]["mean"]
        sigma = likelihoods[c]["std"]

        # Gaussian likelihood: how probable is this score for class c?
        # scipy.stats.norm.pdf computes the Gaussian probability density
        likelihood = norm.pdf(score, loc=mu, scale=sigma)

        # Posterior ∝ prior × likelihood
        posteriors[i, c] = priors[c] * likelihood

    # Normalise so posteriors sum to 1
    total = posteriors[i].sum()
    if total > 0:
        posteriors[i] /= total
    else:
        # Fallback to Agent 2 priors if all likelihoods are zero
        posteriors[i] = priors

    predicted_class[i] = np.argmax(posteriors[i])
    threat_scores[i] = 1 - posteriors[i, benign_idx]

print("Fusion complete.")

# =============================================================================
# STEP 4: EVALUATE AND COMPARE
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Evaluation - Agent 2 alone vs Agent 3 (Bayesian Fusion)")
print("=" * 60)


def compute_far(y_true, y_pred, class_names):
    """False Alarm Rate: fraction of Benign records misclassified as attack."""
    benign_idx = list(class_names).index("Benign")
    benign_mask = (y_true == benign_idx)
    fp = np.sum((y_pred != benign_idx) & benign_mask)
    tn = np.sum((y_pred == benign_idx) & benign_mask)
    return fp / (fp + tn) if (fp + tn) > 0 else 0


def evaluate(name, y_true, y_pred, class_names):
    """Print classification report and key metrics."""
    y_true_names = le.inverse_transform(y_true)
    y_pred_names = le.inverse_transform(y_pred)

    print(f"\n--- {name} ---")
    print(classification_report(y_true_names, y_pred_names, digits=4))

    far = compute_far(y_true, y_pred, class_names)
    print(f"False Alarm Rate : {far:.4f}")
    print(f"Macro F1         : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Accuracy         : {accuracy_score(y_true, y_pred):.4f}")

    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Macro F1": round(f1_score(y_true, y_pred, average='macro'), 4),
        "False Alarm Rate": round(far, 4),
    }


# Agent 2 alone
a2_preds = a2_test["predicted"].values
results_a2 = evaluate("Agent 2 - Random Forest alone", true_labels, a2_preds, class_names)

# Agent 3 - Bayesian fusion
results_a3 = evaluate("Agent 3 - Bayesian Fusion (A1 + A2)", true_labels, predicted_class, class_names)

# Summary comparison table
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
comparison = pd.DataFrame({
    "Agent 2 alone": results_a2,
    "Agent 3 (Bayesian)": results_a3
}).T
print(comparison.to_string())

# Change in FAR - key metric
far_change = results_a3["False Alarm Rate"] - results_a2["False Alarm Rate"]
far_direction = "IMPROVED ↓" if far_change < 0 else "WORSENED ↑"
print(f"\nFAR change: {far_change:+.4f}  ({far_direction})")
f1_change = results_a3["Macro F1"] - results_a2["Macro F1"]
print(f"F1 change:  {f1_change:+.4f}")


# =============================================================================
# STEP 5: CONFUSION MATRICES
# =============================================================================

def plot_cm(y_true, y_pred, title, fname):
    cm = confusion_matrix(le.inverse_transform(y_true),
                          le.inverse_transform(y_pred),
                          labels=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")


print("\nSaving confusion matrices...")
plot_cm(true_labels, a2_preds,
        "Agent 2 - Random Forest alone",
        "cm_agent2.png")
plot_cm(true_labels, predicted_class,
        "Agent 3 - Bayesian Fusion",
        "cm_agent3.png")

# =============================================================================
# STEP 6: THREAT SCORE DISTRIBUTION PLOT
# =============================================================================
# Shows how well threat scores separate Benign from attack traffic.
# Good separation = blue (Benign) peaks near 0, red (attacks) peak near 1.

print("\nSaving threat score distribution...")
benign_mask = (true_labels == benign_idx)
attack_mask = ~benign_mask

plt.figure(figsize=(8, 4))
plt.hist(threat_scores[benign_mask], bins=50, alpha=0.6,
         color='steelblue', label='Benign', density=True)
plt.hist(threat_scores[attack_mask], bins=50, alpha=0.6,
         color='tomato', label='Attack', density=True)
plt.title("Agent 3 - Threat Score Distribution")
plt.xlabel("Threat Score (higher = more likely an attack)")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "threat_score_distribution.png"), dpi=150)
plt.close()
print("Saved: threat_score_distribution.png")

# =============================================================================
# STEP 7: SAVE RESULTS
# =============================================================================

results_df = pd.DataFrame({
    "true_label": true_labels,
    "true_class": le.inverse_transform(true_labels),
    "agent2_predicted": a2_preds,
    "agent3_predicted": predicted_class,
    "threat_score": threat_scores,
    **{f"posterior_{c}": posteriors[:, i]
       for i, c in enumerate(class_names)}
})

results_path = os.path.join(OUTPUT_DIR, "agent3_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\nFull results saved → {results_path}")

comparison.to_csv(os.path.join(OUTPUT_DIR, "agent3_comparison.csv"))
print(f"Comparison table saved → {OUTPUT_DIR}/agent3_comparison.csv")

print("\n" + "=" * 60)
print("AGENT 3 COMPLETE")
print("=" * 60)
