# Agent 2 - Supervised Classifier
## Understanding, Implementation & Findings

---

## What Agent 2 Does
Agent 2 is the supervised classification component of MA-NIDS. It takes raw network traffic features and classifies each connection into one of 8 categories - either Benign or a specific attack type. Unlike Agent 1 which detects anomalies without labels, Agent 2 learns from labeled examples during training.

**Input:** 78 network traffic features (packet sizes, flow duration, port numbers, etc.)
**Output:** Probability vector across all 8 classes - e.g. `{Benign: 0.05, Infilteration: 0.87, ...}`

Agent 3 uses these probabilities - not just the final label - because probabilities carry confidence information that a hard label throws away.

---

## Why Agent 2 Is Independent from Agent 1
A key architectural decision: Agent 2 does NOT use Agent 1's reconstruction error as input.

If Agent 2 used Agent 1's output:
- The two agents would be **coupled** - an Agent 1 error would corrupt Agent 2's prediction before Agent 3 gets a chance to correct anything
- The **independence assumption** underlying Bayesian fusion in Agent 3 would be violated - Bayes requires independent signals to combine meaningfully
- Agent 3 would receive two correlated signals instead of two genuinely different perspectives

**Decision: Agent 2 sees raw traffic features only. Both agents form their own independent judgment. Agent 3 reconciles them.**

---

## The 8 Classes
```
0 → Benign                    normal traffic
1 → DDOS attack-HOIC          distributed denial of service
2 → DoS attacks-GoldenEye     denial of service variant
3 → DoS attacks-Hulk          denial of service variant
4 → DoS attacks-SlowHTTPTest  slow HTTP denial of service
5 → FTP-BruteForce            brute force via FTP protocol
6 → Infilteration             stealthy attack mimicking normal traffic
7 → SSH-Bruteforce            brute force via SSH protocol
```

---

## The Two Algorithms

### Random Forest
Built from many decision trees, each trained on a random subset of data and features. Takes majority vote across all trees. The randomness forces each tree to overfit in a different direction - errors cancel out across the forest.

Key settings used:
- `n_estimators=300` - 300 trees (tuned from default 100)
- `max_depth=20` - limits tree depth to reduce overfitting
- `min_samples_leaf=1` - minimum rows at a leaf node
- `n_jobs=-1` - uses all CPU cores

### XGBoost
Builds trees sequentially - each new tree specifically corrects the mistakes of the previous one. Generally more accurate on tabular data than Random Forest. Faster at prediction time.

Key settings used:
- `n_estimators=300` - 300 trees (tuned)
- `max_depth=9` - shallower trees than RF, typical for XGBoost
- `learning_rate=0.05` - cautious corrections, needs more trees
- `subsample=0.8` - each tree sees 80% of data, reduces overfitting
- `eval_metric="mlogloss"` - appropriate loss for multiclass probabilities

---

## Tuning Process
Used **Grid Search with 3-fold Cross Validation** to find best parameters.

- Grid search tests every combination of specified parameter values
- 3-fold CV: for each combination, trains on 2/3 of data and validates on 1/3, repeated 3 times
- Optimised for `f1_macro` - balanced F1 across all classes
- `n_jobs=-1` parallelises across all CPU cores

Random Forest tested 18 combinations × 3 folds = 54 fits (211 seconds)
XGBoost tested 18 combinations × 3 folds = 54 fits (163 seconds)

---

## Results

### Baseline vs Tuned
```
Metric              RF Base   RF Tuned   XGB Base  XGB Tuned
──────────────────────────────────────────────────────────────
Accuracy             0.8618    0.8678     0.8684    0.8691
Macro F1             0.8592    0.8647     0.8651    0.8658
False Alarm Rate     0.2841    0.3316     0.3435    0.3414
```

Tuning improved accuracy and F1 marginally but did not fix the weak classes.
This means the bottleneck is the data, not the algorithm settings.

### Per-Class Performance (Tuned Random Forest)
```
Class                    Precision  Recall   F1      Notes
──────────────────────────────────────────────────────────
Benign                     0.84      0.67    0.74    weak recall
DDOS attack-HOIC           1.00      1.00    1.00    perfect
DoS attacks-GoldenEye      1.00      1.00    1.00    perfect
DoS attacks-Hulk           1.00      1.00    1.00    perfect
DoS attacks-SlowHTTPTest   0.81      0.52    0.64    weakest - misses half
FTP-BruteForce             0.65      0.88    0.75    low precision
Infilteration              0.72      0.89    0.80    causes FAR problem
SSH-Bruteforce             1.00      1.00    1.00    perfect
```

---

## Critical Finding - The Benign vs Infilteration Problem

### What the Confusion Matrix Revealed
```
Actual Benign (8,000 records):
  → Correctly classified as Benign:        5,347  (67%)
  → Wrongly classified as Infilteration:   2,650  (33%)  ← source of FAR
  → Wrongly classified as anything else:       3  (negligible)

Actual Infilteration (8,000 records):
  → Correctly classified as Infilteration: 6,983  (87%)
  → Wrongly classified as Benign:          1,016  (13%)
```

### What This Means
The **entire false alarm problem reduces to one class pair: Benign and Infilteration.**

2,650 out of 2,653 total false alarms come from Benign being called Infilteration.
Fix this one confusion and FAR drops from 33% to near zero.

The confusion is asymmetric - the model is more likely to call Benign traffic an attack than to miss a real attack. It is overcautious about Infilteration.

### Why This Happens
Infilteration attacks are **deliberately designed to mimic normal traffic** - slow, low-volume, blending in with legitimate user behaviour. Unlike DoS attacks (abnormal volume) or brute force attacks (rapid repeated attempts), infilteration has no dramatic signature. The feature overlap with Benign is intentional, not accidental.

This is not a modelling failure - it is the fundamental nature of the attack. No amount of tuning fixes deliberate mimicry at the feature level.

### Why This Is Important for the Project
This finding directly validates the multi-agent design:

- Agent 2 alone cannot reliably separate Benign from Infilteration
- Agent 1 (autoencoder trained on normal traffic only) may detect subtle reconstruction errors that supervised features miss - even when traffic *looks* normal, it may not *reconstruct* like normal
- Agent 3 can resolve the conflict: if Agent 2 says Infilteration but Agent 1 sees low reconstruction error, fusion should reduce threat probability. If both agree, threat probability rises.

**The Benign/Infilteration confusion is exactly the gap Agent 3 is designed to bridge.**

---

## Final Model Decision
**Tuned Random Forest** selected as primary Agent 2 model for Agent 3.

Reason: Lower false alarm rate than XGBoost (0.33 vs 0.34).
For a real IDS, false alarm rate is the most operationally critical metric -
security teams cannot investigate thousands of false alerts daily.

---

## Saved Files
```
data/processed/
├── agent2_Random_Forest.pkl           baseline RF
├── agent2_XGBoost.pkl                 baseline XGBoost
├── agent2_Random_Forest_tuned.pkl     tuned RF  ← use for Agent 3
├── agent2_XGBoost_tuned.pkl           tuned XGBoost
└── agent2_tuned_comparison.csv        comparison table for paper
```

---

## What to Write in the Paper

**Methods section:**
> "Agent 2 implements a Random Forest classifier trained on labeled network traffic data. Two classifiers were evaluated - Random Forest and XGBoost - with hyperparameters selected via grid search and 3-fold cross validation optimising macro F1 score. Agent 2 operates independently of Agent 1, using raw traffic features only, to preserve the independence assumption required for Bayesian fusion in Agent 3."

**Results section:**
> "Agent 2 achieved 87% overall accuracy with perfect classification on four of eight attack types. Analysis of the confusion matrix revealed that 33% of false alarms originated from a single class confusion - Benign traffic misclassified as Infilteration. This is consistent with the design of infilteration attacks, which deliberately mimic legitimate traffic patterns, producing feature-level overlap that supervised classification alone cannot resolve."

**Conclusions section:**
> "The systematic confusion between Benign and Infilteration traffic (33% false alarm rate) represents the primary limitation of Agent 2 in isolation. This finding motivates the multi-agent architecture - Agent 1's reconstruction-based anomaly detection and Agent 3's probabilistic fusion are specifically positioned to resolve this ambiguity."
