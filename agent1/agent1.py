"""
Agent 1 - Unsupervised Anomaly Detector
Multi-Agent Network Intrusion Detection System (MA-NIDS)
CS 5100: Foundations of Artificial Intelligence

Detects anomalous network traffic using two unsupervised methods:
  1. Autoencoder     - learns to reconstruct normal traffic;
                       high reconstruction error signals an anomaly
  2. Isolation Forest - isolates outliers using random splits;
                       short isolation path signals an anomaly

Both models train ONLY on normal (Benign) traffic.
Uses shared train/test indices from preprocessing so Agent 3
can align Agent 1 and Agent 2 outputs row by row.
Output: agent1_scores.csv - loaded by Agent 3 for Bayesian fusion.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    "data_path": "../data/processed/cleaned_data.csv",
    "train_idx_path": "../data/processed/train_idx.npy",
    "test_idx_path": "../data/processed/test_idx.npy",
    "output_dir": "../data/processed/agent1_outputs",

    # ── Label settings ─────────────────────────────────────────────────────
    "label_column": "Label",
    "normal_class_value": 0,  # 0 = Benign after label encoding

    # ── Autoencoder settings ───────────────────────────────────────────────
    "ae_encoding_dim": 16,  # increased from 8 - less aggressive compression
    "ae_epochs": 150,  # increased from 50 - loss was still falling
    "ae_batch_size": 256,
    "ae_learning_rate": 0.001,

    # ── Isolation Forest settings ──────────────────────────────────────────
    "if_n_estimators": 100,
    "if_contamination": 0.1,  # reduced from 0.4 - was causing too many false alarms

    # ── Output ─────────────────────────────────────────────────────────────
    "save_models": True
}


# =============================================================================
# STEP 1: DATA LOADING
# =============================================================================

def load_and_prepare_data(config):
    """
    Loads data and uses shared indices from preprocessing.
    Ensures Agent 1 evaluates on the exact same test records as Agent 2.
    Applies RobustScaler then clips to [-10, 10] to handle extreme outliers.
    """
    print("\n" + "=" * 60)
    print("STEP 1: Loading and preparing data")
    print("=" * 60)

    df = pd.read_csv(config["data_path"])
    print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    X = df.drop(columns=[config["label_column"]]).values.astype(np.float32)
    y_raw = df[config["label_column"]].values
    y_binary = np.where(y_raw == config["normal_class_value"], 0, 1)

    # Load shared indices - same split used by Agent 2
    train_idx = np.load(config["train_idx_path"])
    test_idx = np.load(config["test_idx_path"])
    print(f"Shared split loaded - Train: {len(train_idx):,}  "
          f"Test: {len(test_idx):,}")

    # Fit scaler on normal training records only
    normal_in_train = train_idx[y_binary[train_idx] == 0]
    scaler = RobustScaler()
    scaler.fit(X[normal_in_train])
    X_scaled = scaler.transform(X)

    # CRITICAL: clip BEFORE slicing into train/test
    # Some features (Idle Min/Max, Active Min/Max) have extreme outlier values
    # even after RobustScaling. Clipping to [-10, 10] keeps MSE loss manageable
    # so the autoencoder can actually learn.
    X_scaled = np.clip(X_scaled, -10, 10)

    print(f"After scaling + clipping to [-10, 10]:")
    print(f"  Min: {X_scaled.min():.4f}  Max: {X_scaled.max():.4f}")
    print(f"  Mean: {X_scaled.mean():.4f}  Std: {X_scaled.std():.4f}")

    # NOW slice into train/test using clipped data
    train_normal_mask = y_binary[train_idx] == 0
    X_train_normal = X_scaled[train_idx[train_normal_mask]]
    X_test = X_scaled[test_idx]
    y_test = y_binary[test_idx]

    print(f"\nTraining set (normal only): {X_train_normal.shape[0]:,}")
    print(f"Test set (mixed)          : {X_test.shape[0]:,}")
    print(f"  - Normal in test        : {np.sum(y_test == 0):,}")
    print(f"  - Anomaly in test       : {np.sum(y_test == 1):,}")

    if config["save_models"]:
        os.makedirs(config["output_dir"], exist_ok=True)
        joblib.dump(scaler,
                    os.path.join(config["output_dir"], "agent1_scaler.pkl"))
        print(f"\nScaler saved → {config['output_dir']}/agent1_scaler.pkl")

    return X_train_normal, X_test, y_test, scaler


# =============================================================================
# STEP 2: AUTOENCODER
# =============================================================================

class Autoencoder(nn.Module):
    """
    Symmetric autoencoder with gradual compression.
    78 → 64 → 32 → 16 (bottleneck) → 32 → 64 → 78

    More gradual compression than 78→32→16→8 gives the encoder
    a better chance to learn meaningful representations.
    Normal traffic reconstructs well (low error).
    Attack traffic does not (high error = anomaly signal).
    """

    def __init__(self, n_features, encoding_dim=16):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, n_features)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_train_normal, config):
    print("\n" + "=" * 60)
    print("STEP 2a: Training Autoencoder")
    print("=" * 60)

    X_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, X_tensor)
    loader = DataLoader(dataset,
                        batch_size=config["ae_batch_size"],
                        shuffle=True)

    n_features = X_train_normal.shape[1]
    model = Autoencoder(n_features, config["ae_encoding_dim"])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["ae_learning_rate"])

    loss_history = []
    print(f"Training for {config['ae_epochs']} epochs...")

    for epoch in range(config["ae_epochs"]):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch + 1:3d}/{config['ae_epochs']}]  "
                  f"Loss: {avg_loss:.6f}")

    print("Autoencoder training complete.")

    if config["save_models"]:
        path = os.path.join(config["output_dir"], "autoencoder.pt")
        torch.save(model.state_dict(), path)
        print(f"Model saved → {path}")

    return model, loss_history


def get_autoencoder_scores(model, X):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        reconstructed = model(X_tensor)
        errors = ((X_tensor - reconstructed) ** 2).mean(dim=1).numpy()
    return errors


def select_ae_threshold(model, X_train_normal, percentile=95):
    train_errors = get_autoencoder_scores(model, X_train_normal)
    threshold = np.percentile(train_errors, percentile)
    print(f"\nAutoencoder threshold (p{percentile}): {threshold:.6f}")
    return threshold


# =============================================================================
# STEP 3: ISOLATION FOREST
# =============================================================================

def train_isolation_forest(X_train_normal, config):
    print("\n" + "=" * 60)
    print("STEP 2b: Training Isolation Forest")
    print("=" * 60)

    iso_forest = IsolationForest(
        n_estimators=config["if_n_estimators"],
        contamination=config["if_contamination"],
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_normal)
    print(f"Isolation Forest trained with {config['if_n_estimators']} trees.")

    if config["save_models"]:
        path = os.path.join(config["output_dir"], "isolation_forest.pkl")
        joblib.dump(iso_forest, path)
        print(f"Model saved → {path}")

    return iso_forest


def get_isolation_forest_scores(iso_forest, X):
    return -iso_forest.score_samples(X)


# =============================================================================
# STEP 4: COMBINE SCORES
# =============================================================================

def normalize_scores(scores):
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def compute_agent1_scores(ae_scores, if_scores, ae_weight=0.5, if_weight=0.5):
    ae_norm = normalize_scores(ae_scores)
    if_norm = normalize_scores(if_scores)
    combined = (ae_weight * ae_norm) + (if_weight * if_norm)
    return combined, ae_norm, if_norm


def select_combined_threshold(combined_scores, y_true):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 100):
        preds = (combined_scores > t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"\nBest combined threshold: {best_t:.3f}  (F1 = {best_f1:.4f})")
    return best_t


def predict_from_scores(combined_scores, threshold=0.5):
    return (combined_scores > threshold).astype(int)


# =============================================================================
# STEP 5: EVALUATION
# =============================================================================

def evaluate_agent1(y_true, y_pred, combined_scores):
    print("\n" + "=" * 60)
    print("EVALUATION: Agent 1 (Combined AE + Isolation Forest)")
    print("=" * 60)

    print(classification_report(
        y_true, y_pred,
        target_names=["Normal (0)", "Anomaly (1)"],
        digits=4
    ))

    try:
        auc = roc_auc_score(y_true, combined_scores)
        print(f"AUC-ROC: {auc:.4f}  (1.0 = perfect, 0.5 = random)")
    except Exception:
        print("AUC-ROC: could not compute")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives  (Normal correctly identified) : {tn:,}")
    print(f"  False Positives (Normal flagged as anomaly)   : {fp:,}  "
          f"← FAR = {far:.4f}")
    print(f"  False Negatives (Anomaly missed)              : {fn:,}")
    print(f"  True Positives  (Anomaly correctly caught)    : {tp:,}")


# =============================================================================
# STEP 6: VISUALIZATIONS
# =============================================================================

def plot_training_loss(loss_history, output_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color='steelblue', linewidth=2)
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ae_training_loss.png"), dpi=150)
    plt.close()


def plot_score_distributions(ae_norm, if_norm, combined, y_true, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sets = [
        (ae_norm, "Autoencoder Score"),
        (if_norm, "Isolation Forest Score"),
        (combined, "Combined Score")
    ]
    for ax, (scores, title) in zip(axes, sets):
        ax.hist(scores[y_true == 0], bins=60, alpha=0.6,
                color='steelblue', label='Normal', density=True)
        ax.hist(scores[y_true == 1], bins=60, alpha=0.6,
                color='tomato', label='Anomaly', density=True)
        ax.set_title(title)
        ax.set_xlabel("Score (higher = more suspicious)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("Agent 1 - Score Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distributions.png"),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title("Agent 1 - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_agent1(config=CONFIG):
    os.makedirs(config["output_dir"], exist_ok=True)

    # Step 1 - Load data using shared indices
    X_train_normal, X_test, y_test, scaler = load_and_prepare_data(config)

    # Step 2a - Autoencoder
    ae_model, loss_history = train_autoencoder(X_train_normal, config)
    _ = select_ae_threshold(ae_model, X_train_normal)
    ae_scores_test = get_autoencoder_scores(ae_model, X_test)

    # Step 2b - Isolation Forest
    iso_forest = train_isolation_forest(X_train_normal, config)
    if_scores_test = get_isolation_forest_scores(iso_forest, X_test)

    # Step 3 - Combine scores
    combined_scores, ae_norm, if_norm = compute_agent1_scores(
        ae_scores_test, if_scores_test
    )

    # Step 4 - Threshold and predict
    best_threshold = select_combined_threshold(combined_scores, y_test)
    y_pred = predict_from_scores(combined_scores, best_threshold)

    # Step 5 - Evaluate
    evaluate_agent1(y_test, y_pred, combined_scores)

    # Step 6 - Plots
    print("\nSaving plots...")
    plot_training_loss(loss_history, config["output_dir"])
    plot_score_distributions(ae_norm, if_norm, combined_scores,
                             y_test, config["output_dir"])
    plot_confusion_matrix(y_test, y_pred, config["output_dir"])
    print("Plots saved.")

    # Save scores for Agent 3
    # Agent 3 loads agent1_combined_score for Bayesian fusion
    scores_df = pd.DataFrame({
        "ae_score_normalized": ae_norm,
        "if_score_normalized": if_norm,
        "agent1_combined_score": combined_scores,
        "agent1_prediction": y_pred,
        "true_label": y_test
    })
    scores_path = os.path.join(config["output_dir"], "agent1_scores.csv")
    scores_df.to_csv(scores_path, index=False)
    print(f"\nAgent 1 scores saved → {scores_path}")

    print("\n" + "=" * 60)
    print("AGENT 1 COMPLETE")
    print("=" * 60)

    return ae_model, iso_forest, combined_scores, y_pred, y_test


if __name__ == "__main__":
    run_agent1(CONFIG)
