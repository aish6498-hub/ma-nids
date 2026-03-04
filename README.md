# MA-NIDS: Multi-Agent Network Intrusion Detection System

A multi-agent network intrusion detection system combining unsupervised anomaly detection, supervised classification, and Bayesian ensemble fusion to detect both known and zero-day attacks.

Built as a final project for CS 5100: Foundations of Artificial Intelligence at Northeastern University.

---

## Overview

Traditional intrusion detection systems struggle with a fundamental tradeoff — supervised methods are accurate on known attacks but blind to novel threats, while unsupervised methods can detect anomalies but produce high false alarm rates. MA-NIDS addresses this by combining both approaches within a multi-agent architecture.

The system consists of three specialized agents:

- **Agent 1 — Anomaly Detector:** An autoencoder trained exclusively on normal traffic. Flags unusual patterns via reconstruction error, enabling detection of zero-day attacks.
- **Agent 2 — Supervised Classifier:** A Random Forest classifier trained on labeled attack data. Identifies and categorizes known attack types with high accuracy.
- **Agent 3 — Bayesian Ensemble:** Fuses the outputs of Agents 1 and 2 using Bayes' Theorem to produce a calibrated posterior threat probability, enabling principled control over the false positive / false negative tradeoff.

---

## Project Structure

```
ma-nids/
│
├── data/                   # Raw and preprocessed datasets (not tracked by git)
│   ├── raw/
│   └── processed/
│
├── agent1/                 # Autoencoder anomaly detector
│   ├── autoencoder.py
│   └── train.py
│
├── agent2/                 # Supervised classifier
│   ├── classifier.py
│   └── train.py
│
├── agent3/                 # Bayesian ensemble fusion
│   └── fusion.py
│
├── evaluation/             # Evaluation metrics and baselines
│   ├── metrics.py
│   └── baselines.py
│
├── notebooks/              # Exploratory analysis and results
│
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Dataset

This project uses the **CSE-CIC-IDS2018** dataset, created jointly by the Communications Security Establishment (CSE) and the Canadian Institute for Cybersecurity (CIC). It contains realistic modern network traffic alongside seven attack scenarios including DoS, DDoS, Botnet, Brute Force, and Web attacks.

The dataset is not included in this repository due to its size. It can be downloaded from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2018.html).

Place downloaded files in `data/raw/` before running any scripts.

---

## Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/ma-nids.git
cd ma-nids
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

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

## References

- Dai, Z., et al. (2024). An intrusion detection model to detect zero-day attacks in unseen data using machine learning. *PLoS ONE*, 19(9), e0308469.
- Soltani, M., et al. (2024). A multi-agent adaptive deep learning framework for online intrusion detection. *Cybersecurity*, 7(1), Article 9.
