# MA-NIDS - Results Analysis
## Full System Understanding & Findings

---

## The Core Claim
The project claimed that combining unsupervised anomaly detection (Agent 1) with supervised classification (Agent 2) through probabilistic fusion (Agent 3) would outperform any single method alone - specifically by reducing false alarm rates while maintaining attack detection accuracy.

**Did it hold up? Yes - partially and meaningfully.**

---

## Agent 1 - Unsupervised Anomaly Detector

### Standalone Performance
```
AUC-ROC:        0.567   (0.5 = random, 1.0 = perfect)
Attack recall:  0.639   catches 64% of attacks
FAR:            0.397   standalone false alarm rate
F1 (anomaly):   0.754
```

### Per-Class Detection Strength
```
Class                    AE Score    vs Benign    Detected?
────────────────────────────────────────────────────────────
SSH-Bruteforce           0.070       7x above     ✓ strong
DoS attacks-GoldenEye    0.059       6x above     ✓ strong
DDOS attack-HOIC         0.022       2x above     ~ moderate
DoS attacks-SlowHTTPTest 0.016       1.8x above   ✗ weak
FTP-BruteForce           0.016       1.8x above   ✗ weak
Infilteration            0.013       1.5x above   ✗ nearly invisible
DoS attacks-Hulk         0.009       BELOW Benign ✗ undetectable
Benign                   0.009       baseline
```

### What This Means
Agent 1 is not a general-purpose detector - it is a specialist. It catches attacks that produce traffic genuinely different from normal. It completely misses attacks designed to blend in with normal traffic.

**Why SSH-Bruteforce and GoldenEye are caught:** These attacks produce rapid repeated connection attempts and unusual packet structures that look very different from normal traffic after scaling. Reconstruction fails - high error - detected.

**Why DoS-Hulk is missed:** Hulk floods with structurally simple HTTP requests. After RobustScaling and clipping, the feature values fall within the range the autoencoder learned from normal traffic. Reconstruction succeeds - low error - undetectable.

**Why Infilteration is missed:** Deliberately designed to mimic legitimate user behaviour. The autoencoder trained on normal traffic cannot distinguish what is intentionally indistinguishable.

**Key insight:** No amount of tuning fixes this. The problem is data overlap at the feature level - a fundamental limitation of reconstruction-based anomaly detection on mimicry attacks.

---

## Agent 2 - Supervised Classifier

### Standalone Performance
```
Accuracy:   86.8%
Macro F1:   0.865
FAR:        33.2%
```

### Per-Class Performance
```
Class                    Precision  Recall   F1      Notes
──────────────────────────────────────────────────────────
Benign                     0.84      0.67    0.74    weak recall
DDOS attack-HOIC           1.00      1.00    1.00    perfect
DoS attacks-GoldenEye      1.00      1.00    1.00    perfect
DoS attacks-Hulk           1.00      1.00    1.00    perfect
DoS attacks-SlowHTTPTest   0.81      0.52    0.64    misses half
FTP-BruteForce             0.65      0.88    0.75    low precision
Infilteration              0.72      0.87    0.79    causes FAR
SSH-Bruteforce             1.00      1.00    1.00    perfect
```

### The Root Cause of 33% FAR
The false alarm problem reduces to **one class pair: Benign vs Infilteration**.

```
Actual Benign (8,000 records):
  → Correctly classified as Benign:        5,347  (67%)
  → Wrongly classified as Infilteration:   2,650  (33%)  ← all false alarms
  → Wrongly classified as anything else:       3  negligible

Actual Infilteration (8,000 records):
  → Correctly classified as Infilteration: 6,983  (87%)
  → Wrongly classified as Benign:          1,016  (13%)
```

2,650 out of 2,653 total false alarms come from Benign being called Infilteration. Fix this one confusion and FAR drops from 33% to near zero.

**Why this happens:** Infilteration deliberately mimics normal traffic. The feature overlap is intentional - supervised classification cannot separate what an adversary designed to look identical.

### What Agent 2 Does Well
Four attack types classified perfectly - DDOS-HOIC, GoldenEye, Hulk, SSH-Bruteforce. These represent the most common volumetric attacks in real networks - DoS floods, DDoS campaigns, brute force attempts.

---

## Agent 3 - Fusion

### Three-Way Final Comparison
```
                  Accuracy  Macro F1  False Alarm Rate
Agent 2 alone       86.8%     0.865            33.2%
Agent 3 Bayesian    86.6%     0.863            31.2%  (-2.0pp)
Agent 3 Stacking    86.7%     0.864            30.5%  (-2.7pp) ✓ best
```

### In Absolute Terms
Stacking produces **217 fewer false alarms per 8,000 normal connections** compared to Agent 2 alone. In a real network handling 100,000 connections per hour, that is approximately 2,700 fewer false alarms per hour.

### Why Stacking Outperforms Bayesian
Bayesian fusion assumes Gaussian likelihoods - a fixed mathematical assumption that doesn't fit all classes equally. When Benign and Infilteration have nearly identical Gaussian distributions, the posterior can only shift as much as the likelihood ratio allows.

Stacking makes no assumptions - it learns the combination rule from data. The meta-model learned to mostly ignore Agent 1's signal for Benign and Infilteration (coefficients near zero) and rely heavily on it for GoldenEye and SSH (coefficients +0.77 and +0.58). This is more adaptive and more accurate.

### What the Meta-Model Learned
```
Class                    Agent 1 coef    What the meta-model learned
────────────────────────────────────────────────────────────────────────
DoS attacks-GoldenEye    +0.778          high score = strong evidence FOR this
SSH-Bruteforce           +0.584          high score = strong evidence FOR this
DoS attacks-Hulk         -0.770          high score = evidence AGAINST this
DDOS attack-HOIC         -0.611          high score = evidence AGAINST this
Benign                   -0.018          Agent 1 barely helps here - ignored
Infilteration            -0.014          Agent 1 barely helps here - ignored
```

The meta-model correctly discovered the same pattern we found manually in Agent 1's per-class analysis. This validates both findings independently.

---

## Does Complementarity Hold?

The original claim was that combining both agents would leverage their complementary strengths.

```
Attack               A1 detects?   A2 detects?   Together
──────────────────────────────────────────────────────────
SSH-Bruteforce       ✓ strong      ✓ perfect      ✓ perfect
DoS-GoldenEye        ✓ strong      ✓ perfect      ✓ perfect
DDOS-HOIC            ✗ weak        ✓ perfect      ✓ A2 carries it
DoS-Hulk             ✗ invisible   ✓ perfect      ✓ A2 carries it
Infilteration        ✗ invisible   ✗ 33% FAR      ~ slight improvement
SlowHTTPTest         ✗ weak        ✗ 52% recall   ~ no improvement
FTP-BruteForce       ✗ weak        ~ 65% prec.    ~ no improvement
```

**Complementarity works for easy cases.** When one agent struggles, the other compensates. Where both agents struggle - Infilteration, SlowHTTPTest, FTP-BruteForce - fusion provides only marginal help because neither agent has a reliable signal to contribute.

---

## Why the Improvement Is Modest - The Honest Answer

The 2-3% FAR improvement is real but not dramatic. Three reasons:

**1. Agent 1 cannot distinguish Benign from Infilteration.**
Mean scores are 0.110 vs 0.120 - nearly identical. Fusion can only correct as much as the signal allows. When Agent 1 sees both classes as identical, it has nothing useful to contribute for that confusion.

**2. The problem is fundamentally hard.**
Infilteration attacks are designed by adversaries to evade detection. Feature overlap is intentional. This is not a modelling failure - it is the nature of the attack.

**3. Agent 1's AUC-ROC is 0.567.**
A stronger anomaly detector would give Agent 3 better signals and produce larger improvements. This is the primary future work direction.

---

## The Most Important Single Result

```
Agent 2 alone (supervised only):    FAR = 33.2%
Agent 3 Stacking (multi-agent):      FAR = 30.5%
Absolute improvement:                    2.7pp
```

This directly validates the core hypothesis - multi-agent fusion reduces false alarms compared to supervised classification alone.

---

## System Strengths

- Perfect detection on 4 of 8 attack types (DDOS-HOIC, GoldenEye, Hulk, SSH)
- Reproducible - full pipeline runs in 187 seconds from raw data
- Two fusion approaches implemented and compared (Bayesian + Stacking)
- Principled architecture - independent agents, shared evaluation indices
- All findings are explainable and connected to the data

## System Limitations

- 30.5% false alarm rate is still high for real deployment
- SlowHTTPTest recall of 52% means half of these attacks are missed
- Agent 1 AUC-ROC of 0.567 limits fusion improvement ceiling
- Infilteration/Benign confusion is fundamental - not fixable with current features
- Tested on one dataset (CIC-IDS2018) - generalisation to other networks unknown

## Future Work - Prioritised by Impact

### The Bottleneck
Almost all remaining improvement opportunity lies in the Benign/Infilteration confusion. 2,650 out of 2,653 false alarms come from this one pair. Everything else is near-perfect. Future work must target this specific problem.

The key question is whether the ceiling is a model problem or a data problem. Infilteration is deliberately designed to look like Benign at the feature level - mean Agent 1 scores are 0.110 vs 0.120, nearly identical. This overlap is intentional by the attacker, not an artifact of model choices. That said, genuine improvements are possible.

---

### Agent 1 - High Impact

**LSTM Autoencoder** ← highest potential improvement
Current autoencoder treats each record as an independent snapshot. LSTM captures temporal patterns across consecutive connections. Infilteration may blend in at the single-record level but reveal itself over time through unusual connection sequences. Directly mentioned as future work in Ahmad et al.

**Variational Autoencoder (VAE)**
Compresses to a probability distribution instead of a fixed point. Produces smoother latent space and often better anomaly scores. Requires replacing bottleneck with mean/variance outputs and adding KL divergence to the loss. Moderate implementation complexity.

**Better feature handling for timing features**
Idle Min/Max, Active Min/Max had extreme outlier values even after RobustScaling - required clipping to [-10, 10]. These features may be more noise than signal for the autoencoder. Dropping or engineering them specifically could improve reconstruction quality.

---

### Agent 2 - Medium Impact

**Class-specific decision thresholds**
Instead of one threshold (50%) for all classes, tune separate thresholds per class. For Infilteration specifically, raising the threshold to 70-80% - only classify as Infilteration if confidence exceeds this - would directly reduce Benign being miscalled as Infilteration.

**Deep Neural Network instead of Random Forest**
A DNN with dropout regularisation can learn non-linear feature interactions that Random Forest misses. Dataset size (256,000 training records) is sufficient. Validated by Dai et al. Tradeoff: longer training and more hyperparameter sensitivity.

**Feature engineering for Infilteration**
Some features that separate Benign from Infilteration may not be in the current 78-feature set:
- Connection duration patterns over time
- Inter-arrival time variance
- Ratio of inbound to outbound bytes

---

### Agent 3 - Medium Impact

**Add XGBoost as second base model in stacking** ← highest ROI, lowest effort
XGBoost test predictions are already saved (`agent2_XGBoost_test_predictions.csv`). Adding those 8 probability columns to stacking meta-features costs ~10 lines of code. Gives the meta-model a second independent supervised signal. Could push FAR below 30%.

**Neural network meta-model**
Logistic Regression is linear - it can only learn linear combinations of agent outputs. A small neural network meta-model could learn non-linear interactions. For example: "when Agent 2 is 70% confident of Infilteration AND Agent 1 score is below 0.05, strongly suggest Benign." LR cannot capture this.

**Per-class threshold on threat score**
Tune separate threat score thresholds per class instead of using argmax(posteriors). Gives direct control over precision/recall tradeoff for each attack type independently.

---

### Parameter Tuning That Would Actually Help

```
High impact:
  Agent 1 - ae_epochs: 150 → 300
            Loss was still declining at epoch 150 - more training would help
  Agent 1 - encoding_dim: 16 → 32
            More bottleneck capacity = richer representations
  Agent 2 - Infilteration threshold: 0.5 → 0.75
            Directly targets the false alarm source

Medium impact:
  Agent 1 - ae_weight: 0.5 → 0.7 (reduce Isolation Forest contribution)
            IF adds noise for subtle attacks - down-weight it
  Agent 3 - stacking C: 1.0 → 0.1 (stronger regularisation)

Low impact:
  Agent 2 - n_estimators beyond 300 (diminishing returns after ~200)
  Agent 1 - batch size changes
```

---

### Generalisation

- Tested on one dataset (CIC-IDS2018) - performance on other networks unknown
- Dataset is from 2018 - modern attack patterns may differ significantly
- Real-time deployment evaluation not conducted
- Testing on CIC-IDS2017, UNSW-NB15 would validate generalisation

---

### What to Say in the Paper

> "The primary limitation of the current system is Agent 1's inability to distinguish Infilteration from Benign traffic at the feature level. The most promising improvement is replacing the feed-forward autoencoder with an LSTM autoencoder that captures temporal patterns across connection sequences - Infilteration attacks may mimic individual connections but reveal themselves over time through unusual sequences of activity. Additionally, incorporating XGBoost as a second base model in the stacking meta-learner would provide a more diverse set of signals for Agent 3 to combine, potentially pushing the false alarm rate below 30%. Finally, class-specific decision thresholds for Infilteration classification in Agent 2 represent a low-cost, high-impact modification that directly targets the source of 99.9% of the system's false alarms."
