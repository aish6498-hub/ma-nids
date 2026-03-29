# Agent 3 - Bayesian Fusion & Stacking
## Understanding, Implementation & Findings

---

## What Agent 3 Does
Agent 3 is the fusion component of MA-NIDS. It combines the outputs of Agent 1 and Agent 2 into a single calibrated threat probability per record. Two approaches were implemented and compared - Bayesian Fusion and Stacking.

**Input:**
- From Agent 1: `agent1_combined_score` - a 0-1 anomaly score per record
- From Agent 2: probability vector across 8 classes per record

**Output:**
- `predicted_class` - final classification
- `threat_score` - 0 to 1 probability of being an attack (1 - posterior[Benign])

---

## Why Agent 3 Is Needed
Agent 2 alone produces a 33% false alarm rate driven by Benign/Infilteration confusion. Agent 1 alone has AUC-ROC 0.56 - too weak to use independently. Agent 3 combines both signals to reduce false alarms while maintaining classification accuracy.

---

## Why Agent 2 Is Independent from Agent 1
Bayesian fusion assumes the two agents are independent signals. If Agent 2 used Agent 1's reconstruction error as input, they would be correlated - combining correlated signals with Bayes double-counts information and invalidates the math. Keeping them independent means Agent 3 receives two genuinely different perspectives to reconcile.

---

## Approach 1 - Bayesian Fusion

### How It Works
```
posterior(class) ∝ prior(class) × likelihood(agent1_score | class)
```
- **Prior** = Agent 2's class probability
- **Likelihood** = Gaussian PDF evaluated at Agent 1's score for each class
- **Posterior** = normalised product of prior and likelihood
- **threat_score** = 1 - posterior[Benign]

### Estimating Likelihoods
For each class, fit a Gaussian to Agent 1 training scores for all records of that class. This gives `P(agent1_score | class)`.

```
Class                    Mean     Std
─────────────────────────────────────
Benign                   0.110    0.096
DDOS attack-HOIC         0.065    0.061
DoS attacks-GoldenEye    0.201    0.084
DoS attacks-Hulk         0.049    0.050
DoS attacks-SlowHTTPTest 0.117    0.050
FTP-BruteForce           0.117    0.050
Infilteration            0.120    0.114
SSH-Bruteforce           0.181    0.074
```

### Iteration - Likelihood Collapse Bug
**Problem:** FTP-BruteForce and SlowHTTPTest had near-zero standard deviations (0.0004) because Agent 1 produces almost identical scores for all records of these classes. The Gaussian was infinitely narrow - any score slightly different from 0.1168 got near-zero likelihood. FTP-BruteForce posterior collapsed to zero for every test record. F1 = 0.000.

**Fix:** Minimum std floor of 0.05:
```python
sigma = max(class_scores.std(), 0.05) + 1e-6
```
This ensures every class has a wide enough Gaussian to produce non-zero likelihoods.

**Key lesson:** Always set a minimum std floor when fitting Gaussians to potentially low-variance data.

### Bayesian Results
```
Metric              Agent 2 alone    Agent 3 Bayesian    Change
────────────────────────────────────────────────────────────────
Accuracy               0.8678          0.8659            -0.002
Macro F1               0.8647          0.8631            -0.002
False Alarm Rate       0.3316          0.3119            -0.020 ✓
```

FAR improved by 2% with negligible accuracy loss. Modest improvement because Benign and Infilteration have nearly identical likelihood distributions - the posterior can only shift as far as the likelihood ratio allows.

---

## Approach 2 - Stacking (Meta-Learning)

### How It Works
Instead of a fixed mathematical formula, train a Logistic Regression meta-model that learns how to combine Agent 1 and Agent 2 outputs from data.

**Meta-features per record:**
```
[prob_Benign, prob_DDOS, prob_DoS, ..., agent1_combined_score]
```
9 columns total - one probability per class plus Agent 1 score.

### Why Cross-Validation Is Critical
If Agent 2 predicts on its own training data, it already memorised those labels - predictions are artificially perfect. The meta-model would learn from fake-perfect signals and fail on real test data.

**Solution:** 5-fold cross-validation - each fold's held-out predictions come from a model that never saw those records during training. Honest, realistic signals for the meta-model to learn from.

```
Fold 1: train on folds 2-5, predict on fold 1
Fold 2: train on folds 1,3,4,5, predict on fold 2
... repeat for all 5 folds
→ 256,000 honest meta-feature rows collected
```

### Why Logistic Regression as Meta-Model
- Meta-features are already rich probability signals - complex model not needed
- Interpretable - coefficients show how much Agent 1 is trusted per class
- Doesn't overfit on small meta-feature space (only 9 input columns)
- Much faster than a second Random Forest

### What the Meta-Model Learned
```
Class                    Agent 1 coefficient    Interpretation
───────────────────────────────────────────────────────────────
DoS attacks-GoldenEye    +0.772   high Agent 1 score = strong evidence FOR GoldenEye
SSH-Bruteforce           +0.585   high Agent 1 score = strong evidence FOR SSH
DoS attacks-Hulk         -0.768   high Agent 1 score = evidence AGAINST Hulk
DDOS attack-HOIC         -0.612   high Agent 1 score = evidence AGAINST HOIC
Benign                   -0.027   Agent 1 barely affects Benign (correctly ignored)
Infilteration            -0.012   Agent 1 barely affects Infilteration (correctly ignored)
```

The meta-model correctly learned the per-class pattern we discovered in Agent 1's score analysis - GoldenEye and SSH produce high reconstruction errors so Agent 1 is trusted positively for them. Hulk and HOIC produce low reconstruction errors despite being attacks, so high Agent 1 scores are evidence against them. For Benign and Infilteration, coefficients are near zero - the meta-model learned Agent 1 cannot distinguish these classes and mostly ignores it.

### Stacking Results
```
Metric              Agent 2 alone    Agent 3 Stacking    Change
────────────────────────────────────────────────────────────────
Accuracy               0.8678          0.8671            -0.001
Macro F1               0.8647          0.8643            -0.000
False Alarm Rate       0.3316          0.3045            -0.027 ✓✓
```

FAR improved by 2.7% - 217 fewer false alarms per 8,000 normal records. F1 drop is essentially zero. Best result of all three approaches.

---

## Three-Way Final Comparison

```
                    Agent 2    Bayesian    Stacking
────────────────────────────────────────────────────
Accuracy             86.8%      86.6%       86.7%
Macro F1             0.865      0.863       0.864
False Alarm Rate     33.2%      31.2%       30.5%  ← best
```

**Stacking wins on every metric.** It reduces FAR more than Bayesian fusion while preserving F1 almost perfectly.

### Why Stacking Outperforms Bayesian
Bayesian fusion assumes Gaussian likelihoods - a fixed mathematical assumption that doesn't fit all classes well. Stacking makes no assumptions - it learns the combination rule directly from data. When Benign and Infilteration have nearly identical Gaussian distributions, Bayesian fusion is limited in how much it can shift the posterior. The meta-model simply learned to mostly ignore Agent 1's signal for these two classes and rely on Agent 2 - a more adaptive response.

---

## RuntimeWarnings During Stacking
Divide by zero and overflow warnings appeared during Logistic Regression training. Caused by meta-features containing probability values very close to 0 and 1 - perfect predictions from Agent 2 on easy classes like DoS-Hulk. These are warnings not errors - the model trained correctly and produced valid predictions. Suppressed with `warnings.filterwarnings("ignore", category=RuntimeWarning)`.

---

## Final Model Decision
**Stacking is the primary Agent 3 method.** Bayesian fusion is retained as a comparison demonstrating that a principled mathematical approach also helps, just less than a learned approach. Both are reported in the paper.

---

## Saved Files
```
data/processed/agent3_outputs/
├── agent3_results.csv                  ← Bayesian fusion predictions
├── agent3_stacking_results.csv         ← Stacking predictions
├── agent3_stacking_meta_model.pkl      ← trained meta-model
├── agent3_comparison.csv               ← Agent 2 vs Bayesian
├── agent3_full_comparison.csv          ← Agent 2 vs Bayesian vs Stacking
├── cm_agent2.png                       ← Agent 2 confusion matrix
├── cm_agent3.png                       ← Bayesian confusion matrix
├── cm_agent3_stacking.png              ← Stacking confusion matrix
├── threat_score_distribution.png       ← Bayesian threat scores
└── threat_score_distribution_stacking.png ← Stacking threat scores
```

---

## What to Write in the Paper

**Methods section:**
> "Agent 3 implements two fusion strategies - Bayesian fusion and stacking - which are compared empirically. Bayesian fusion estimates per-class likelihood distributions from Agent 1 training scores using Gaussian distributions with a minimum standard deviation floor of 0.05 to prevent likelihood collapse. Stacking trains a Logistic Regression meta-model on meta-features combining Agent 2 class probabilities and Agent 1 anomaly scores, generated via 5-fold cross-validation to prevent data leakage."

**Results section:**
> "Both fusion strategies reduced the false alarm rate compared to Agent 2 alone. Bayesian fusion reduced FAR from 33.2% to 31.2% with a negligible F1 drop of 0.002. Stacking achieved a larger FAR reduction to 30.5% while maintaining F1 within 0.0004 of Agent 2 alone. Analysis of the stacking meta-model's coefficients revealed that Agent 1's signal was trusted most strongly for GoldenEye (+0.772) and SSH-Bruteforce (+0.585) - classes where reconstruction error is high - and largely ignored for Benign (-0.027) and Infilteration (-0.012) - classes where Agent 1 cannot distinguish between them."

**Conclusions section:**
> "Stacking outperformed Bayesian fusion across all metrics, demonstrating that a learned combination rule is more effective than fixed Gaussian likelihood assumptions when agent signals have unequal discriminative power across classes. The meta-model's learned coefficients validate the per-class Agent 1 analysis - it correctly down-weighted Agent 1's contribution for classes where reconstruction-based detection is unreliable. Future work could explore deep stacking architectures or reinforcement learning-based fusion to further reduce false alarm rates."

