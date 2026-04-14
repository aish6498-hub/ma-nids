"""
Agent 2 - Hyperparameter Tuning (offline, run once)

Finds optimal hyperparameters for Random Forest and XGBoost using Grid Search with 3-fold Cross Validation,
optimising for macro F1 to ensure balanced performance across all 8 traffic classes.

Best parameters found and hardcoded into agent2/train.py:
  Random Forest : max_depth=20, min_samples_leaf=1, n_estimators=300
  XGBoost       : learning_rate=0.05, max_depth=9, n_estimators=300, subsample=0.8

This script does not need to be rerun - it is preserved for reproducibility so the tuning decisions
can be inspected and verified.
"""

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score,
                             f1_score, confusion_matrix)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Configuration

DATA_PATH = "../data/processed/cleaned_data.csv"
ENCODER_PATH = "../data/processed/label_encoder.pkl"
OUTPUT_DIR = "../data/processed/"
LABEL_COL = "Label"
SEED = 42

# Load Data

print("Loading data...")
df = pd.read_csv(DATA_PATH)
le = joblib.load(ENCODER_PATH)

X = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL]

train_idx = np.load("../data/processed/train_idx.npy")
test_idx = np.load("../data/processed/test_idx.npy")

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

# Parameter Grids

# Each key is a parameter name, each value is a list of options to try
# Grid search tests every combination:
#   Random Forest : 3 (n_estimators) × 3 (max_depth) × 2 (min_samples_leaf) = 18 combinations
#   XGBoost       : 3 (n_estimators) × 3 (max_depth) × 2 (learning_rate) × 1 (subsample) = 18 combinations
# With 3-fold CV: 18 × 3 = 54 training runs per model

rf_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1, 4]
}

xgb_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 6, 9],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8]
}


# Tuning Function

def tune_and_evaluate(name, base_model, param_grid, X_train, X_test, y_train, y_test, le):
    print(f"\n{'=' * 55}")
    print(f"  Tuning: {name}")
    print('=' * 55)

    # GridSearchCV tests every parameter combination
    # cv=3 means 3-fold cross validation
    # scoring='f1_macro' optimises for balanced performance across all classes
    # refit=True means it automatically retrains on full training data using the best parameters found
    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    t_start = time.time()
    search.fit(X_train, y_train)
    tune_time = time.time() - t_start

    print(f"\nBest parameters: {search.best_params_}")
    print(f"Best CV F1 score: {search.best_score_:.4f}")
    print(f"Tuning time: {tune_time:.1f}s")

    # Evaluate best model on held-out test set
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)

    print(f"\nClassification Report (best {name}):")
    print(classification_report(y_test_labels, y_pred_labels))

    # False alarm rate
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)
    benign_idx = list(le.classes_).index("Benign")
    tn = cm[benign_idx, benign_idx]
    fp = cm[benign_idx, :].sum() - tn
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"False Alarm Rate: {far:.4f}")

    # Save best model
    path = f"{OUTPUT_DIR}/agent2_{name.replace(' ', '_')}_tuned.pkl"
    joblib.dump(best_model, path)
    print(f"Tuned model saved to {path}")

    return {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Macro F1": round(f1_score(y_test, y_pred, average='macro'), 4),
        "False Alarm Rate": round(far, 4),
        "Best Params": search.best_params_,
        "Tune Time (s)": round(tune_time, 1)
    }


# Run Tuning

results = {}

results["Random Forest"] = tune_and_evaluate(
    "Random Forest",
    RandomForestClassifier(random_state=SEED, n_jobs=-1),
    rf_grid,
    X_train, X_test, y_train, y_test, le
)

results["XGBoost"] = tune_and_evaluate(
    "XGBoost",
    XGBClassifier(random_state=SEED, n_jobs=-1,
                  eval_metric="mlogloss", verbosity=0),
    xgb_grid,
    X_train, X_test, y_train, y_test, le
)

# Comparison Table

print(f"\n{'=' * 55}")
print("  TUNED MODEL COMPARISON")
print('=' * 55)

summary = {k: {m: v for m, v in r.items() if m != "Best Params"}
           for k, r in results.items()}
comparison = pd.DataFrame(summary).T
print(comparison.to_string())

print("\nBest Parameters:")
for name, r in results.items():
    print(f"  {name}: {r['Best Params']}")

comparison.to_csv(f"{OUTPUT_DIR}/agent2_tuned_comparison.csv")
print(f"\nSaved to {OUTPUT_DIR}/agent2_tuned_comparison.csv")
