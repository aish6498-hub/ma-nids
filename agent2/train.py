"""
Agent 2 - Supervised Classifier

Trains two supervised classifiers on labeled network traffic data:
  1. Random Forest  - ensemble of decision trees, selected as primary Agent 2 model
  2. XGBoost        - gradient boosted trees, trained as second base model for stacking

Both classifiers learn to distinguish between 8 traffic classes:
  Benign, DDOS-HOIC, DoS-GoldenEye, DoS-Hulk, DoS-SlowHTTPTest, FTP-BruteForce, Infilteration, SSH-BruteForce

Design decisions:
  - Uses shared train/test indices from preprocessing (train_idx.npy / test_idx.npy)
    so all three agents evaluate on identical test records - required for Agent 3 to join outputs row by row
  - Saves per-class probabilities (not just predicted labels) because Agent 3
    needs soft probability vectors, not hard decisions, for both Bayesian fusion and stacking meta-learning
  - Hyperparameters are tuned values from agent2/tune.py (grid search, 3-fold CV)

Outputs:
  - agent2_Random_Forest.pkl                    : trained RF model
  - agent2_XGBoost.pkl                          : trained XGBoost model
  - agent2_Random_Forest_test_predictions.csv   : RF predictions + probabilities
  - agent2_XGBoost_test_predictions.csv         : XGBoost predictions + probabilities
  - agent2_comparison.csv                       : side-by-side metrics summary
"""

import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from xgboost import XGBClassifier

# Configuration

DATA_PATH = "../data/processed/cleaned_data.csv"
ENCODER_PATH = "../data/processed/label_encoder.pkl"
TRAIN_IDX = "../data/processed/train_idx.npy"
TEST_IDX = "../data/processed/test_idx.npy"
OUTPUT_DIR = "../data/processed"
LABEL_COL = "Label"
SEED = 42

# Load Data

print("Loading data...")
df = pd.read_csv(DATA_PATH)
le = joblib.load(ENCODER_PATH)

X = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL]

# Load shared indices - same split used by all agents.
# Critical: ensures Agent 3 can join Agent 1 and Agent 2 outputs row by row.
train_idx = np.load(TRAIN_IDX)
test_idx = np.load(TEST_IDX)

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

print(f"Features : {X.shape[1]}")
print(f"Train    : {len(X_train):,}")
print(f"Test     : {len(X_test):,}")
print(f"Classes  : {list(le.classes_)}")

# Define Models

models = {
    "Random_Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=SEED
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.8,
        n_jobs=-1,
        random_state=SEED,
        eval_metric="mlogloss",
        verbosity=0
    )
}

# Train, Evaluate, Save

results = {}

for name, model in models.items():
    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print('=' * 55)

    # Train
    t_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t_start
    print(f"Training time: {train_time:.1f}s")

    # Predict labels and probabilities
    t_start = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    predict_time = time.time() - t_start
    print(f"Prediction time: {predict_time:.2f}s")

    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels))

    # False alarm rate
    cm = confusion_matrix(y_test_labels, y_pred_labels,
                          labels=le.classes_)
    benign_idx = list(le.classes_).index("Benign")
    tn = cm[benign_idx, benign_idx]
    fp = cm[benign_idx, :].sum() - tn
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"False Alarm Rate: {far:.4f}")

    # Confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_{name}.png")
    plt.close()

    # Save model
    model_path = f"{OUTPUT_DIR}/agent2_{name}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved → {model_path}")

    # Save test predictions and probabilities for Agent 3
    # Agent 3 needs:
    #   - true_label    : ground truth for evaluation
    #   - predicted     : argmax class label (used by Bayesian fusion)
    #   - prob_*        : one probability column per class (used as meta-features in stacking fusion)
    proba_cols = {f"prob_{cls}": y_pred_proba[:, i]
                  for i, cls in enumerate(le.classes_)}
    preds_df = pd.DataFrame({
        "true_label": y_test.values,
        "predicted": y_pred,
        **proba_cols
    })
    preds_path = f"{OUTPUT_DIR}/agent2_{name}_test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"Test predictions saved → {preds_path}")

    # Store summary metrics
    results[name] = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Macro F1": round(f1_score(y_test, y_pred,
                                   average='macro'), 4),
        "False Alarm Rate": round(far, 4),
        "Train Time (s)": round(train_time, 1),
        "Predict Time (s)": round(predict_time, 2)
    }

# Comparison Table

print(f"\n{'=' * 55}")
print("  FINAL COMPARISON")
print('=' * 55)
comparison = pd.DataFrame(results).T
print(comparison.to_string())
comparison.to_csv(f"{OUTPUT_DIR}/agent2_comparison.csv")
print(f"\nComparison saved → {OUTPUT_DIR}/agent2_comparison.csv")
print("\nNote: using tuned hyperparameters (RF: max_depth=20, n=300 | "
      "XGB: lr=0.05, depth=9, n=300)")
