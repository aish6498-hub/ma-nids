# MA-NIDS: Multi-Agent Network Intrusion Detection System

A multi-agent system for network intrusion detection that combines unsupervised anomaly detection, supervised
classification, and probabilistic fusion to detect both known attacks and novel zero-day threats.

---

## The Problem

Traditional intrusion detection systems face a fundamental tradeoff:

- **Supervised methods** are accurate on known attack types but blind to novel threats they have never seen before
- **Unsupervised methods** can detect anomalies without labels but produce high false alarm rates

MA-NIDS addresses this by running both approaches in parallel through independent agents and combining their outputs
through a learned fusion layer.

---

## Architecture

The system consists of three independent agents, each receiving the same raw network traffic features (78 columns). No
agent sees another agent's output during training - independence is intentional and required for valid probabilistic
fusion.

```
Raw Network Traffic (78 features)
            │
            ├─────────────────────────┐
            ▼                         ▼
        Agent 1                   Agent 2
     Autoencoder               Random Forest
    (unsupervised)              + XGBoost
    trains on Benign           (supervised)
    traffic only               trains on all
                                8 classes
            │                         │
     anomaly score            probability vector
     (1 value,                (8 values, one
     0–1 scale)                 per class)
            │                         │
            └───────────┬─────────────┘
                        ▼
                    Agent 3
                 Bayesian Fusion
                        OR
                    Stacking ← primary approach
                        │
                        ▼
              Final threat assessment
              (predicted class + threat score)
```

### Agent 1 - Unsupervised Anomaly Detector

An autoencoder neural network trained **exclusively on normal (Benign) traffic**. It learns to reconstruct normal
traffic patterns. At inference time:

- Normal traffic → reconstructs well → low error → classified as Benign
- Attack traffic → reconstructs poorly → high error → flagged as anomaly

Architecture: `78 → 128 → 64 → 32 → 16 → 32 → 64 → 128 → 78` with BatchNorm layers. An Isolation Forest is also trained
but contributes zero weight to the final score as the autoencoder alone produces better class separation on this dataset.

### Agent 2 - Supervised Classifier

A Random Forest classifier (primary) and XGBoost (second base model for stacking), both trained on labeled traffic data
across all 8 attack classes. Hyperparameters tuned via grid search with 3-fold cross-validation optimising macro F1.

### Agent 3 - Fusion Layer

Two fusion approaches implemented and compared:

**Bayesian Fusion** - Treats Agent 2 probabilities as the prior and Agent 1 scores as the likelihood (modelled as
per-class Gaussians). Computes posterior via Bayes' theorem. In practice this produces no improvement because the
autoencoder scores are too uniformly low across all classes for the Gaussians to carry meaningful signal.

**Stacking (primary)** - A Logistic Regression meta-model trained via 5-fold cross-validation that learns the
combination rule from data. Meta-features: RF probabilities (8) + XGBoost probabilities (8) + Agent 1 score (1) = 17
columns. Makes no distributional assumptions and adapts to signal quality per class automatically.

### Agent 2 Threshold Tuning

A class-specific confidence threshold for the Infilteration class. Since 99.9% of all false alarms come from Benign
traffic being misclassified as Infilteration, raising the prediction threshold for that class from 50% to 63% cuts the
false alarm rate nearly in half.

---

## Results

### Full Comparison

| Method                     | Accuracy   | Macro F1   | False Alarm Rate |
|----------------------------|------------|------------|------------------|
| Agent 1 alone (binary)     | 88.52%     | -          | 46.19%           |
| Agent 2 alone (RF)         | 86.78%     | 0.8647     | 33.16%           |
| Agent 3 - Bayesian Fusion  | 86.78%     | 0.8647     | 33.18%           |
| **Agent 3 - Stacking**     | **86.89%** | **0.8660** | **30.30%**       |
| Agent 2 + Threshold (0.63) | 85.77%     | 0.8544     | 16.15%           |

Stacking reduces false alarms by **2.86 percentage points** over supervised classification alone - equivalent to
approximately 226 fewer false alarms per 8,000 normal connections - while simultaneously improving F1 and accuracy.

### Per-Class Performance

| Attack Class             | Precision | Recall | F1   | A2 Stacking F1 | Notes                             |
|--------------------------|-----------|--------|------|----------------|-----------------------------------|
| Benign                   | 0.84      | 0.67   | 0.74 | 0.76           | ↑ improved by stacking            |
| DDOS attack-HOIC         | 1.00      | 1.00   | 1.00 | 1.00           | perfect - both agents agree       |
| DoS attacks-GoldenEye    | 1.00      | 1.00   | 1.00 | 1.00           | perfect - Agent 1 strong signal   |
| DoS attacks-Hulk         | 1.00      | 1.00   | 1.00 | 1.00           | perfect - Agent 2 carries it      |
| DoS attacks-SlowHTTPTest | 0.81      | 0.52   | 0.64 | 0.64           | no improvement - both agents weak |
| FTP-BruteForce           | 0.65      | 0.88   | 0.75 | 0.75           | no improvement - both agents weak |
| Infilteration            | 0.72      | 0.87   | 0.79 | 0.79           | causes 99.9% of false alarms      |
| SSH-Bruteforce           | 1.00      | 1.00   | 1.00 | 1.00           | perfect - Agent 1 strong signal   |

### Root Cause of False Alarms

Nearly all false alarms in this system reduce to a single class confusion:
Benign traffic being misclassified as Infilteration.

|                                  | Predicted Benign | Predicted Infilteration          | Predicted Other |
|----------------------------------|------------------|----------------------------------|-----------------|
| **Actual Benign (8,000)**        | 5,347 (66.8%)    | 2,650 (33.1%) ← all false alarms | 3 (0.0%)        |
| **Actual Infilteration (8,000)** | 1,017 (12.7%)    | 6,983 (87.3%)                    | 0 (0.0%)        |

2,650 of 2,653 total false alarms - **99.9%** - come from this one confusion.
Fix this single pair and FAR drops from 33.2% to near zero.

**Why this happens:** Infilteration attacks are deliberately engineered to mimic
normal user behaviour. The feature overlap between Benign and Infilteration is
intentional as it is the attack's evasion strategy and not considered a modelling failure.
Neither Agent 1 (anomaly scores nearly identical for both classes) nor Agent 2
(supervised classifier cannot separate what an adversary designed to look
identical) can reliably resolve this confusion. This is a fundamental limitation
of the current feature set and cannot be fixed by tuning.

### Threshold Tuning Tradeoff

The default argmax rule predicts Infilteration whenever its probability exceeds
any other class - even a 42% confidence prediction is treated identically to a
95% prediction. A class-specific threshold says: only predict Infilteration if
confidence exceeds X%, otherwise fall back to Benign.

The threshold was tuned via 5-fold CV on training data (test set never touched)
by scanning 61 values from 0.30 to 0.90, selecting the best FAR while keeping
F1 drop under 1%.

| Threshold      | FAR        | Macro F1   | Notes                         |
|----------------|------------|------------|-------------------------------|
| 0.50 (default) | 33.16%     | 0.8647     | baseline - barely helps       |
| 0.55           | 28.09%     | 0.8626     | meaningful drop begins        |
| 0.60           | 20.76%     | 0.8580     | aggressive territory          |
| **0.63**       | **16.15%** | **0.8544** | **best tradeoff - selected**  |
| 0.70           | 10.27%     | 0.8432     | too aggressive                |
| 0.80           | 4.86%      | 0.8234     | misses too many Infilteration |

**What the threshold buys:**

- 1,700 fewer false alarms per 8,000 normal connections
- Cost: 2,009 more missed Infilteration attacks (recall drops 87.3% → 62.2%)

This is the fundamental precision/recall tradeoff made explicit and
operator-controllable. A security team can choose their operating point
based on whether false alarms or missed attacks are more costly in their
environment - the threshold gives principled control over that decision.

---

## Dataset

**CSE-CIC-IDS2018** - Created by the Canadian Institute for Cybersecurity. Contains realistic network traffic alongside
seven attack scenarios including DoS, DDoS, Brute Force, and Infilteration attacks.

The dataset is not included in this repository due to size (~4GB raw). Download from
the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2018.html) and place files in `data/raw/`
before running.

**Preprocessing decisions:**

- Excluded `02-20-2018.csv` - 84 columns vs 80 in all other files (Src Port mismatch)
- Dropped 5 rare/unreliable classes (Brute Force Web/XSS, DDOS-LOIC-UDP, SQL Injection, Slowloris)
- Capped each class at 40,000 samples for balance
- Final dataset: 320,000 records × 78 features, 8 balanced classes
- 80/20 stratified train/test split with shared indices across all agents

---

## Project Structure

```
ma-nids/
│
├── data/
│   ├── raw/                         # Raw CIC-IDS2018 CSV files (not tracked)
│   ├── processed/                   # Cleaned data and model outputs
│   └── preprocess.py                # Cleaning, balancing, shared split
│
├── agent1/
│   ├── agent1.py                    # Autoencoder training + anomaly scoring
│   └── save_train_scores.py         # Computes training scores for Agent 3
│
├── agent2/
│   ├── train.py                     # Random Forest + XGBoost training
│   ├── threshold_tuning.py          # Infilteration threshold tuning
│   └── tune.py                      # Hyperparameter tuning (run once)
│
├── agent3/
│   ├── fusion.py                    # Bayesian fusion
│   └── stacking.py                  # Stacking meta-model (primary approach)
│
├── run_pipeline.py                  # Runs full 7-step pipeline in order
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline from scratch (preprocessing through fusion)
python3 run_pipeline.py

# Skip preprocessing - data already cleaned
python3 run_pipeline.py --from 2

# Skip to Agent 2 training only
python3 run_pipeline.py --from 4

# Run Agent 3 Bayesian fusion only
python3 run_pipeline.py --from 6

# Run Agent 3 Stacking only
python3 run_pipeline.py --from 7
```

**Total runtime:** ~267 seconds from scratch. Step 7 (Stacking) uses a cached meta-training set on repeat runs - reduces
to ~4 seconds after the first run.

> ⚠️ Delete `data/processed/agent3_outputs/meta_X_train_cache.npy` and `meta_y_train_cache.npy` if Agent 1 or Agent 2 is
> retrained, as cached meta-features will no longer match the updated models.

---

## Pipeline Steps

| Step | Script                        | Description                                 | Runtime                     |
|------|-------------------------------|---------------------------------------------|-----------------------------|
| 1    | `data/preprocess.py`          | Clean, balance, encode, save shared indices | ~30s                        |
| 2    | `agent1/agent1.py`            | Train autoencoder on normal traffic         | ~80s                        |
| 3    | `agent1/save_train_scores.py` | Compute training scores for Agent 3         | ~3s                         |
| 4    | `agent2/train.py`             | Train Random Forest + XGBoost               | ~25s                        |
| 5    | `agent2/threshold_tuning.py`  | Tune Infilteration threshold via CV         | ~51s                        |
| 6    | `agent3/fusion.py`            | Bayesian fusion evaluation                  | ~13s                        |
| 7    | `agent3/stacking.py`          | Stacking meta-model training + evaluation   | ~95s first run / ~4s cached |

---

## Requirements

- Python 3.10+
- torch >= 2.0.0
- scikit-learn >= 1.3.0
- xgboost >= 2.1.4
- scipy >= 1.11.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- joblib >= 1.3.0

See `requirements.txt` for full pinned versions.

---

## Key Findings

**What works well:**

- Four of eight attack types detected with perfect F1 (DDOS-HOIC, GoldenEye, Hulk, SSH-Bruteforce)
- Stacking fusion improves all three metrics simultaneously over Agent 2 alone
- Threshold tuning cuts FAR nearly in half with modest F1 cost
- Bayesian and Stacking fusion both implemented, compared, and documented with honest analysis

**What doesn't work and why:**

- Bayesian fusion produces no improvement - the improved autoencoder scores are too uniformly low across all classes for
  per-class Gaussians to carry signal
- Infilteration/Benign confusion accounts for 99.9% of false alarms and is not fixable with current features - the
  attack is designed to mimic normal traffic
- DoS-SlowHTTPTest recall of 52% - feature overlap with normal HTTP traffic
- Agent 1 standalone FAR of 46% makes it unsuitable as a standalone detector

**Why our results differ from published benchmarks:**
Published results on CIC-IDS2018 typically report 95–99% accuracy, but these are commonly evaluated on imbalanced data
with potential feature leakage. Our balanced 8-class evaluation (40,000 samples per class) is a harder and more
realistic benchmark.

---

## Future Work

- LSTM autoencoder to capture temporal traffic patterns (would improve Agent 1 AUC-ROC beyond its current 0.8733)
- Feature engineering to better separate Infilteration from Benign
- Neural network meta-model for deeper stacking
- Cross-dataset evaluation for generalisation testing
- Real-time deployment with streaming data

---

## References

- Dai, Z., et al. (2024). An intrusion detection model to detect zero-day attacks in unseen data using machine learning.
  *PLoS ONE*, 19(9), e0308469. https://doi.org/10.1371/journal.pone.0308469
- Soltani, M., et al. (2024). A multi-agent adaptive deep learning framework for online intrusion detection.
  *Cybersecurity*, 7(1), Article 9. https://doi.org/10.1186/s42400-023-00199-0
- Ahmad, Z., et al. (2018). Network intrusion detection system: A systematic study of machine learning and deep learning
  approaches. *Transactions on Emerging Telecommunications Technologies*, 32(1), e4150.
