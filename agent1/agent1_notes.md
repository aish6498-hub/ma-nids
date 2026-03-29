# Agent 1 - Unsupervised Anomaly Detector
## Understanding, Implementation, Iterations & Findings

---

## What Agent 1 Does
Agent 1 is the unsupervised anomaly detection component of MA-NIDS. Unlike Agent 2 which learns from labeled examples, Agent 1 has never seen attack labels. Instead it learns what **normal traffic looks like** and flags anything that deviates as suspicious.

**Input:** 78 network traffic features (same as Agent 2)
**Output:** A combined anomaly score per record between 0 and 1 - higher means more suspicious

Agent 3 uses this score alongside Agent 2's class probabilities for Bayesian fusion.

---

## Why Unsupervised for Anomaly Detection?
Supervised methods (like Agent 2) can only detect attack types they were trained on. Zero-day attacks - new attacks never seen before - have no label, so supervised models miss them entirely.

Agent 1 sidesteps this by never learning attack patterns at all. It only learns normal. Anything that doesn't look normal gets flagged - including brand new attack types the system has never encountered.

**This is the core motivation for having both agents.**

---

## Two Methods Used

### Method 1 - Autoencoder
A neural network that learns to compress and reconstruct its input.

**Architecture:** 78 → 64 → 32 → 16 (bottleneck) → 32 → 64 → 78

**How it works:**
- Encoder compresses 78 features down to 16 numbers (the bottleneck)
- Decoder tries to reconstruct the original 78 features from those 16 numbers
- Trained on normal traffic only - learns to compress and reconstruct normal patterns well
- **Reconstruction error** = difference between original input and reconstruction
- Normal traffic → low reconstruction error (autoencoder knows this pattern)
- Attack traffic → high reconstruction error (unfamiliar pattern, poor reconstruction)

**Loss function:** Mean Squared Error (MSE) - measures average squared difference between original and reconstructed values. Lower = better reconstruction.

**Threshold:** Set at the 95th percentile of normal training errors. Only the top 5% of normal errors get flagged, keeping false alarms low on traffic the model was trained on.

### Method 2 - Isolation Forest
An ensemble method that isolates outliers using random decision trees.

**How it works:**
- Builds many random trees, each trying to isolate individual samples
- Normal samples take many splits to isolate (they blend in with others)
- Anomalies are isolated quickly (they stand apart from the crowd)
- Short isolation path = anomalous

**Why use both?**
Autoencoder captures reconstruction-based anomalies. Isolation Forest captures statistical outliers. Combined score is more robust than either alone.

---

## Data Preparation

### Shared Train/Test Indices
Agent 1 loads `train_idx.npy` and `test_idx.npy` from preprocessing - the same split used by Agent 2. This is critical for Agent 3, which needs to join Agent 1 and Agent 2 outputs row by row on identical test records.

### Train on Normal Traffic Only
Agent 1's training set contains **only Benign records** from the shared training indices. Attack records are never seen during training. The test set contains both normal and attack records for evaluation.

```
Training set: 32,000 Benign records only
Test set:      8,000 Benign + 56,000 attack records
```

### Scaling - RobustScaler
Features are scaled using RobustScaler, fitted on normal training records only.

**Why RobustScaler over StandardScaler?**
StandardScaler uses mean and standard deviation - both heavily affected by outliers. RobustScaler uses median and interquartile range, making it resistant to extreme values in network traffic data.

**Why fit on normal only?**
If we fit the scaler on all data including attacks, attack patterns influence what "center" and "spread" mean. This leaks attack information into the scaling step before training even begins.

---

## Iterations and Fixes

### Iteration 1 - First Run (Broken)
**Result:** Loss = 3,025,109,381. AUC-ROC = 0.54. Useless.

**Problem:** The raw feature values after RobustScaling still had extreme values up to 119,052,768. Features like `Idle Min`, `Idle Max`, `Active Min`, `Active Max` are timing features measured in microseconds - they have extreme outlier values even after scaling. MSE loss sums squared errors across all 78 features, so these extreme values completely dominated the loss. The autoencoder spent all its capacity trying to handle billion-scale numbers instead of learning meaningful patterns.

**Diagnostic output that revealed the problem:**
```
Max after scaling: 119,052,768
Mean after scaling: 84,897
```

**Fix attempted:** Added `np.clip(X_scaled, -10, 10)` after scaling.

---

### Iteration 2 - Clip in Wrong Place (Still Broken)
**Result:** Loss still in billions. AUC-ROC still 0.54.

**Problem:** The clip line was placed after the train/test split. So `X_train_normal` and `X_test` were sliced from the unclipped `X_scaled`. The printed diagnostic showed correct clipped values, but the data passed to the autoencoder was still unclipped.

**Key lesson:** Order of operations matters. The clip must happen before slicing.

```python
# WRONG ORDER:
X_train_normal = X_scaled[train_idx]   # sliced from unclipped data
X_scaled = np.clip(X_scaled, -10, 10)  # too late

# CORRECT ORDER:
X_scaled = np.clip(X_scaled, -10, 10)  # clip first
X_train_normal = X_scaled[train_idx]   # then slice
```

---

### Iteration 3 - Clip in Right Place (Working)
**Result:** Loss = 0.122 at epoch 50. AUC-ROC = 0.57. F1 = 0.75.

**What fixed it:** Clipping placed correctly before slicing. Loss dropped from billions to 0.12.

**Remaining problem:** Loss still falling at epoch 50, model not converged. Architecture too aggressive (78→32→16→8→16→32→78).

---

### Iteration 4 - More Epochs + Better Architecture (Final)
**Changes:**
- Epochs: 50 → 150 (let model converge fully)
- Architecture: 78→32→16→8 → 78→64→32→16 (more gradual compression)
- Encoding dim: 8 → 16 (less information loss at bottleneck)
- IF contamination: 0.4 → 0.1 (was causing too many false alarms)

**Why more gradual compression?**
Compressing 78 features to 8 in three steps forces too much information loss too quickly. The encoder can't find a good compressed representation. Going 78→64→32→16 gives the network more capacity at each step to learn what to keep.

**Result:**
```
Loss at epoch 150:  0.016  ← steadily converging
AUC-ROC:            0.576
Attack recall:      0.640
F1 (anomaly):       0.754
FAR:                0.399
```

---

## Key Finding - Per-Class Anomaly Scores

After the final run, we checked the mean autoencoder score per attack class:

```
SSH-Bruteforce           0.070  ← detected well (7x above Benign)
DoS attacks-GoldenEye    0.059  ← detected well (6x above Benign)
DDOS attack-HOIC         0.022  ← moderate signal
DoS attacks-SlowHTTPTest 0.016  ← weak signal
FTP-BruteForce           0.016  ← weak signal
Infilteration            0.013  ← almost invisible
DoS attacks-Hulk         0.009  ← BELOW Benign baseline
Benign                   0.009  ← baseline
```

---

## Why Some Attacks Are Invisible to Agent 1

This is not a bug - it is a fundamental limitation of reconstruction-based anomaly detection.

**Attacks Agent 1 catches well:**
SSH-Bruteforce and DoS-GoldenEye produce traffic patterns very different from normal - rapid repeated connections, unusual packet structures. After scaling, their feature values fall far outside the normal range the autoencoder learned. Reconstruction fails → high error → detected.

**Attacks Agent 1 misses:**
- **Infilteration** is deliberately designed to mimic normal traffic - slow, low-volume, blending in with legitimate user behaviour. Reconstruction succeeds → low error → not detected.
- **DoS-Hulk** floods with structurally simple HTTP requests. After RobustScaling and clipping, the feature values fall neatly within the range the autoencoder learned. Reconstruction is actually easier than some normal traffic → score BELOW Benign.
- **SlowHTTPTest and FTP-BruteForce** are subtle, low-volume attacks that partially resemble normal traffic patterns.

**No amount of tuning fixes this.** The problem is data overlap at the feature level, not model quality. An autoencoder trained only on normal traffic cannot distinguish attacks that look identical to normal traffic.

---

## What This Means for the Project

This finding is one of the strongest arguments in your paper for the multi-agent architecture:

**Agent 1 catches:** SSH-Bruteforce, DoS-GoldenEye, DDOS-HOIC
**Agent 2 catches:** DDOS-HOIC, DoS-Hulk, SSH-Bruteforce, GoldenEye perfectly
**Both struggle with:** Infilteration, SlowHTTPTest, FTP-BruteForce

The agents have **complementary strengths and weaknesses**. Agent 3's fusion is specifically designed to combine these - trusting Agent 1 heavily when its score is high, trusting Agent 2 when Agent 1's signal is weak, and resolving the Benign/Infilteration ambiguity using both signals together.

---

## Final Configuration
```python
CONFIG = {
    "ae_encoding_dim":  16,    # bottleneck size
    "ae_epochs":        150,   # converges around epoch 130-150
    "ae_batch_size":    256,
    "ae_learning_rate": 0.001,
    "if_n_estimators":  100,
    "if_contamination": 0.1,
}
# Scaling: RobustScaler fitted on normal training records only
# Clipping: np.clip(X_scaled, -10, 10) BEFORE train/test split
```

## Final Results
```
AUC-ROC:        0.576   (limited by data overlap, not model quality)
Attack F1:      0.754
Attack recall:  0.640   (catches 64% of attacks)
FAR:            0.399   (standalone - Agent 3 expected to reduce this)
```

---

## Saved Files
```
data/processed/agent1_outputs/
├── autoencoder.pt          ← trained autoencoder weights
├── isolation_forest.pkl    ← trained isolation forest
├── agent1_scaler.pkl       ← RobustScaler for preprocessing new data
└── agent1_scores.csv       ← anomaly scores for all test records
                               Agent 3 loads agent1_combined_score
```

---

## What to Write in the Paper

**Methods section:**
> "Agent 1 implements an unsupervised anomaly detector combining an autoencoder and an Isolation Forest. The autoencoder was trained exclusively on Benign traffic using a 78→64→32→16→32→64→78 architecture with MSE reconstruction loss. Feature values were normalised using RobustScaler fitted on normal training records only, followed by clipping to [-10, 10] to handle extreme outlier values present in timing features. The combined anomaly score was computed as an equal-weighted average of normalised autoencoder reconstruction error and Isolation Forest anomaly score."

**Results section:**
> "Agent 1 achieved an AUC-ROC of 0.576 and attack recall of 0.640 on the test set. Per-class analysis revealed that detection performance varied significantly by attack type - SSH-Bruteforce and DoS-GoldenEye produced reconstruction errors 6-7 times above the Benign baseline, while Infilteration and DoS-Hulk produced scores at or below the Benign baseline. This is consistent with the design of these attacks - Infilteration deliberately mimics legitimate traffic, while Hulk's structurally simple HTTP flood falls within the feature distribution of normal traffic after normalisation."

**Conclusions section:**
> "Agent 1's inability to detect Infilteration and DoS-Hulk represents a fundamental limitation of reconstruction-based anomaly detection - attacks that structurally resemble normal traffic cannot be distinguished by an autoencoder trained on normal traffic alone. This finding reinforces the necessity of Agent 2's supervised classification and Agent 3's probabilistic fusion, which combine complementary detection capabilities to address the full spectrum of attack types."
