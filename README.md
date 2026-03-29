# MA-NIDS: Multi-Agent Network Intrusion Detection System

A multi-agent network intrusion detection system combining unsupervised anomaly detection, supervised classification, and probabilistic fusion to detect both known and zero-day attacks.

Built as a final project for CS 5100: Foundations of Artificial Intelligence at Northeastern University.

---

## Results

| Method | Accuracy | Macro F1 | False Alarm Rate |
|--------|----------|----------|-----------------|
| Agent 2 alone (Random Forest) | 86.8% | 0.865 | 33.2% |
| Agent 3 - Bayesian Fusion | 86.6% | 0.863 | 31.2% |
| Agent 3 - Stacking | **86.7%** | **0.864** | **30.5%** |

Stacking reduces false alarms by 2.7 percentage points over supervised classification alone - equivalent to 217 fewer false alarms per 8,000 normal connections - while maintaining accuracy within 0.1%.

Four of eight attack types are detected with perfect F1 scores (DDOS-HOIC, DoS-GoldenEye, DoS-Hulk, SSH-Bruteforce). The primary limitation is Infilteration traffic, which is deliberately designed to mimic normal behaviour and accounts for nearly all false alarms.

---

## Overview

Traditional intrusion detection systems struggle with a fundamental tradeoff - supervised methods are accurate on known attacks but blind to novel threats, while unsupervised methods can detect anomalies but produce high false alarm rates. MA-NIDS addresses this by combining both approaches within a multi-agent architecture.

The system consists of three specialized agents:

- **Agent 1 - Anomaly Detector:** An autoencoder and Isolation Forest trained exclusively on normal traffic. Flags unusual patterns via reconstruction error, enabling detection of zero-day attacks.
- **Agent 2 - Supervised Classifier:** A Random Forest classifier trained on labeled attack data. Identifies and categorizes known attack types with high accuracy.
- **Agent 3 - Fusion:** Combines Agent 1 and Agent 2 outputs using two approaches - Bayesian fusion (Gaussian likelihood estimation) and Stacking (Logistic Regression meta-model trained via 5-fold cross-validation). Stacking is the primary approach.

---

## Project Structure

```
ma-nids/
│
├── data/                        # Raw and preprocessed datasets (not tracked by git)
│   ├── raw/
│   ├── processed/
    └── preprocess.py            # Data cleaning, balancing, shared split
│
├── agent1/                      # Anomaly detector
│   ├── agent1.py                # Autoencoder + Isolation Forest training
│   └── save_train_scores.py     # Saves training scores for Agent 3
│
├── agent2/                      # Supervised classifier
│   ├── train.py                 # Random Forest + XGBoost training
    └── agent2_tune.py           # Hyperparameter tuning (run once, results hardcoded)
│
├── agent3/                      # Fusion layer
│   ├── fusion.py                # Bayesian fusion
│   └── stacking.py              # Stacking meta-model (primary)
│
├── requirements.txt             # Python dependencies
├── run_pipeline.py              # Runs full pipeline in order
└── README.md
```

---

## Dataset

This project uses the **CSE-CIC-IDS2018** dataset, created jointly by the Communications Security Establishment (CSE) and the Canadian Institute for Cybersecurity (CIC). It contains realistic modern network traffic alongside seven attack scenarios including DoS, DDoS, Botnet, Brute Force, and Web attacks.

The dataset is not included in this repository due to its size. It can be downloaded from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2018.html).

Place downloaded files in `data/raw/` before running any scripts.

---

## Quick Start
```bash
# Run full pipeline from scratch
python3 run_pipeline.py

# Skip preprocessing (data already cleaned)
python3 run_pipeline.py --from 2

# Run only fusion (agents already trained)
python3 run_pipeline.py --from 5
```

Total runtime: ~187 seconds on a standard laptop.

---

## Requirements

Key dependencies:
- Python 3.10+
- scikit-learn
- PyTorch
- pandas
- numpy
- matplotlib
- seaborn

See `requirements.txt` for full list.

---

## Team

| Name | Component |
|------|-----------|
| Person 1 | Data preprocessing & Agent 1 (Autoencoder + Isolation Forest) |
| Person 2 | Agent 2 (Supervised Classifier) & Agent 3 (Fusion) |
| Person 3 | Agent 3 (Stacking) & Evaluation |

---

## References

- Dai, Z., et al. (2024). An intrusion detection model to detect zero-day attacks in unseen data using machine learning. *PLoS ONE*, 19(9), e0308469.
- Soltani, M., et al. (2024). A multi-agent adaptive deep learning framework for online intrusion detection. *Cybersecurity*, 7(1), Article 9.
- Ahmad, Z., et al. (2021). Network intrusion detection system: A systematic study of machine learning and deep learning approaches. *Transactions on Emerging Telecommunications Technologies*, 32(1), e4150.