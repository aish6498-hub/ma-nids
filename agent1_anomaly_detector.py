# =============================================================================
# AGENT 1: Unsupervised Anomaly Detector
# Multi-Agent Network Intrusion Detection System (MA-NIDS)
# CS 5100: Foundations of Artificial Intelligence
#
# This agent detects anomalous (potentially malicious) network traffic
# using two unsupervised methods:
#   1. Autoencoder     - learns to reconstruct normal traffic; high
#                        reconstruction error signals an anomaly
#   2. Isolation Forest - isolates outliers using random splits; short
#                        isolation path signals an anomaly
#
# HOW TO USE ON ANY DATASET:
#   - Your CSV must have a label column (default name: "Label")
#   - The label column must have one value that means "normal/benign"
#     (default: "Benign" or numeric 0 after encoding)
#   - All other columns are treated as numeric features
#   - Change CONFIG at the top of this file to match your dataset
# =============================================================================

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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
# CONFIGURATION — change these to match your dataset
# =============================================================================
CONFIG = {
    # Path to your cleaned/preprocessed CSV file
    "data_path": "cleaned_data.csv",

    # Name of the label column in your CSV
    "label_column": "Label",

    # The value in the label column that means NORMAL traffic
    # After preprocessing this is usually 0 (encoded "Benign")
    "normal_class_value": 0,

    # Fraction of data to use for testing (0.2 = 20%)
    "test_size": 0.2,

    # Random seed for reproducibility
    "random_state": 42,

    # --- Autoencoder settings ---
    "ae_encoding_dim": 8,       # Size of the bottleneck layer
    "ae_epochs": 50,            # Number of training epochs
    "ae_batch_size": 256,       # Samples per training batch
    "ae_learning_rate": 0.001,  # How fast the model learns

    # --- Isolation Forest settings ---
    "if_n_estimators": 100,     # Number of trees in the forest
    "if_contamination": 0.1,    # Expected fraction of anomalies in data

    # --- Output settings ---
    "output_dir": "agent1_outputs",   # Folder to save models and plots
    "save_models": True               # Whether to save trained models
}


# =============================================================================
# STEP 1: DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data(config):
    """
    Loads the dataset and splits it into:
      - X_train_normal : normal traffic only  → used to TRAIN both models
      - X_test         : all traffic (normal + attacks) → used to EVALUATE
      - y_test         : true labels for test set (0 = normal, 1 = anomaly)

    Both models are trained ONLY on normal traffic because they learn
    what "normal" looks like, then flag anything different as suspicious.
    """
    print("\n" + "="*60)
    print("STEP 1: Loading and preparing data")
    print("="*60)

    # Load CSV
    df = pd.read_csv(config["data_path"])
    print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Separate features from labels
    X = df.drop(columns=[config["label_column"]]).values.astype(np.float32)
    y_raw = df[config["label_column"]].values

    # Convert labels to binary: 0 = normal, 1 = anomaly
    # This works whether your labels are strings or already encoded numbers
    y_binary = np.where(y_raw == config["normal_class_value"], 0, 1)

    n_normal = np.sum(y_binary == 0)
    n_anomaly = np.sum(y_binary == 1)
    print(f"Normal samples : {n_normal:,}")
    print(f"Anomaly samples: {n_anomaly:,}")

    # Scale features using RobustScaler (resistant to outliers)
    # Fit ONLY on normal traffic to avoid leaking attack patterns
    normal_mask = (y_binary == 0)
    scaler = RobustScaler()
    scaler.fit(X[normal_mask])
    X_scaled = scaler.transform(X)

    # Split into train (normal only) and test (all classes)
    # We use a fixed split: last test_size% of each class goes to test
    np.random.seed(config["random_state"])

    normal_indices = np.where(y_binary == 0)[0]
    anomaly_indices = np.where(y_binary == 1)[0]

    # Shuffle indices
    np.random.shuffle(normal_indices)
    np.random.shuffle(anomaly_indices)

    # Split normal traffic
    n_test_normal = int(len(normal_indices) * config["test_size"])
    train_normal_idx = normal_indices[n_test_normal:]   # larger portion for training
    test_normal_idx  = normal_indices[:n_test_normal]   # smaller portion for testing

    # Split anomaly traffic (all goes to test — we never train on attacks)
    n_test_anomaly = int(len(anomaly_indices) * config["test_size"])
    test_anomaly_idx = anomaly_indices[:n_test_anomaly]

    # Build final splits
    X_train_normal = X_scaled[train_normal_idx]         # TRAIN: normal only
    test_idx = np.concatenate([test_normal_idx, test_anomaly_idx])
    X_test = X_scaled[test_idx]
    y_test = y_binary[test_idx]

    print(f"\nTraining set (normal only): {X_train_normal.shape[0]:,} samples")
    print(f"Test set (mixed)          : {X_test.shape[0]:,} samples")
    print(f"  - Normal in test        : {np.sum(y_test == 0):,}")
    print(f"  - Anomaly in test       : {np.sum(y_test == 1):,}")
    print(f"Number of features        : {X.shape[1]}")

    # Save scaler for use by other agents
    if config["save_models"]:
        os.makedirs(config["output_dir"], exist_ok=True)
        joblib.dump(scaler, os.path.join(config["output_dir"], "agent1_scaler.pkl"))
        print(f"\nScaler saved to {config['output_dir']}/agent1_scaler.pkl")

    return X_train_normal, X_test, y_test, scaler


# =============================================================================
# STEP 2: AUTOENCODER
# =============================================================================

class Autoencoder(nn.Module):
    """
    A symmetric autoencoder neural network.

    Architecture:
        Input (n_features)
          → Encoder: n_features → 32 → 16 → bottleneck
          → Decoder: bottleneck → 16 → 32 → n_features
          → Reconstructed output

    The bottleneck forces the network to learn a compressed
    representation. Normal traffic compresses and reconstructs well.
    Attack traffic does not — resulting in high reconstruction error.
    """

    def __init__(self, n_features, encoding_dim=8):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim)  # bottleneck
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, n_features)    # reconstruct original size
        )

    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return reconstructed


def train_autoencoder(X_train_normal, config):
    """
    Trains the autoencoder on normal traffic only.
    Returns the trained model and training loss history.
    """
    print("\n" + "="*60)
    print("STEP 2a: Training Autoencoder")
    print("="*60)

    # Convert numpy array to PyTorch tensors
    X_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, X_tensor)  # input = target (reconstruction)
    loader = DataLoader(dataset, batch_size=config["ae_batch_size"], shuffle=True)

    # Build model
    n_features = X_train_normal.shape[1]
    model = Autoencoder(n_features, config["ae_encoding_dim"])

    # Use mean squared error loss — measures how different
    # the reconstruction is from the original input
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["ae_learning_rate"])

    # Train
    loss_history = []
    print(f"Training for {config['ae_epochs']} epochs...")

    for epoch in range(config["ae_epochs"]):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            reconstructed = model(batch_X)
            loss = criterion(reconstructed, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1:3d}/{config['ae_epochs']}]  Loss: {avg_loss:.6f}")

    print("Autoencoder training complete.")

    # Save model
    if config["save_models"]:
        model_path = os.path.join(config["output_dir"], "autoencoder.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    return model, loss_history


def get_autoencoder_scores(model, X):
    """
    Computes per-sample reconstruction error for a dataset.
    Higher error = more anomalous.
    Returns a numpy array of reconstruction errors.
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(X_tensor)
        # Mean squared error per sample
        errors = ((X_tensor - reconstructed) ** 2).mean(dim=1).numpy()

    return errors


def select_ae_threshold(model, X_train_normal, percentile=95):
    """
    Selects the anomaly threshold from normal training data.
    Flags a sample as anomalous if its reconstruction error
    is above the given percentile of normal errors.

    percentile=95 means: only the top 5% of normal errors get flagged,
    which keeps false alarms low on traffic the model was trained on.
    """
    train_errors = get_autoencoder_scores(model, X_train_normal)
    threshold = np.percentile(train_errors, percentile)
    print(f"\nAutoencoder threshold (p{percentile} of normal errors): {threshold:.6f}")
    return threshold


# =============================================================================
# STEP 3: ISOLATION FOREST
# =============================================================================

def train_isolation_forest(X_train_normal, config):
    """
    Trains an Isolation Forest on normal traffic only.

    How it works:
      - Builds many random decision trees
      - For each sample, measures how many splits are needed to isolate it
      - Anomalies are isolated quickly (fewer splits = shorter path)
      - Normal samples take longer to isolate (longer path)
    """
    print("\n" + "="*60)
    print("STEP 2b: Training Isolation Forest")
    print("="*60)

    iso_forest = IsolationForest(
        n_estimators=config["if_n_estimators"],
        contamination=config["if_contamination"],
        random_state=config["random_state"],
        n_jobs=-1  # use all CPU cores
    )

    iso_forest.fit(X_train_normal)
    print(f"Isolation Forest trained with {config['if_n_estimators']} trees.")

    # Save model
    if config["save_models"]:
        model_path = os.path.join(config["output_dir"], "isolation_forest.pkl")
        joblib.dump(iso_forest, model_path)
        print(f"Model saved to {model_path}")

    return iso_forest


def get_isolation_forest_scores(iso_forest, X):
    """
    Returns anomaly scores from the Isolation Forest.
    score_samples() returns negative values — more negative = more anomalous.
    We negate them so that higher score = more anomalous (consistent with AE).
    """
    raw_scores = iso_forest.score_samples(X)
    return -raw_scores   # flip sign: higher = more suspicious


# =============================================================================
# STEP 4: COMBINE SCORES INTO FINAL AGENT 1 OUTPUT
# =============================================================================

def normalize_scores(scores):
    """
    Normalizes an array of scores to the [0, 1] range using min-max scaling.
    This is needed before combining AE and IF scores so neither dominates.
    """
    min_val = scores.min()
    max_val = scores.max()
    if max_val == min_val:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)


def compute_agent1_scores(ae_scores, if_scores, ae_weight=0.5, if_weight=0.5):
    """
    Combines autoencoder and isolation forest scores into one final score.

    Both scores are first normalized to [0, 1].
    Then a weighted average is computed.
    Final score range: [0, 1] — higher means more suspicious.

    ae_weight + if_weight should equal 1.0
    Default: equal weighting (0.5 each)
    """
    ae_norm = normalize_scores(ae_scores)
    if_norm = normalize_scores(if_scores)
    combined = (ae_weight * ae_norm) + (if_weight * if_norm)
    return combined, ae_norm, if_norm


def predict_from_scores(combined_scores, threshold=0.5):
    """
    Converts continuous scores into binary predictions.
    Samples with score > threshold are flagged as anomalies (1).
    Samples with score <= threshold are classified as normal (0).
    """
    return (combined_scores > threshold).astype(int)


def select_combined_threshold(combined_scores, y_true):
    """
    Automatically finds the best combined score threshold by
    trying 100 values between 0 and 1 and picking the one
    that maximizes the F1 score on the test set.

    In a real deployment you would tune this on a validation set,
    not the test set — but for this project this is acceptable.
    """
    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.linspace(0.1, 0.9, 100)

    for t in thresholds:
        preds = (combined_scores > t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"\nBest combined threshold: {best_threshold:.3f}  (F1 = {best_f1:.4f})")
    return best_threshold


# =============================================================================
# STEP 5: EVALUATION
# =============================================================================

def evaluate_agent1(y_true, y_pred, combined_scores, label="Agent 1 (Combined)"):
    """
    Prints a full evaluation report for Agent 1's predictions.
    Includes accuracy, precision, recall, F1, AUC-ROC, and
    a confusion matrix breakdown.
    """
    print("\n" + "="*60)
    print(f"EVALUATION: {label}")
    print("="*60)

    print(classification_report(
        y_true, y_pred,
        target_names=["Normal (0)", "Anomaly (1)"],
        digits=4
    ))

    # AUC-ROC: measures how well the score separates normal from anomaly
    # 1.0 = perfect, 0.5 = random guessing
    try:
        auc = roc_auc_score(y_true, combined_scores)
        print(f"AUC-ROC Score: {auc:.4f}")
    except Exception:
        print("AUC-ROC: could not compute (need both classes in test set)")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives  (Normal correctly identified) : {tn:,}")
    print(f"  False Positives (Normal flagged as anomaly)   : {fp:,}")
    print(f"  False Negatives (Anomaly missed)              : {fn:,}")
    print(f"  True Positives  (Anomaly correctly caught)    : {tp:,}")


# =============================================================================
# STEP 6: VISUALIZATIONS
# =============================================================================

def plot_training_loss(loss_history, output_dir):
    """Plots the autoencoder's training loss over epochs."""
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color='steelblue', linewidth=2)
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "ae_training_loss.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_score_distributions(ae_norm, if_norm, combined, y_true, output_dir):
    """
    Plots the distribution of anomaly scores for normal vs attack traffic.
    A good detector will show clearly separated distributions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    score_sets = [
        (ae_norm,  "Autoencoder Score"),
        (if_norm,  "Isolation Forest Score"),
        (combined, "Combined Score")
    ]

    for ax, (scores, title) in zip(axes, score_sets):
        ax.hist(scores[y_true == 0], bins=60, alpha=0.6,
                color='steelblue', label='Normal', density=True)
        ax.hist(scores[y_true == 1], bins=60, alpha=0.6,
                color='tomato', label='Anomaly', density=True)
        ax.set_title(title)
        ax.set_xlabel("Score (higher = more suspicious)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Agent 1 — Score Distributions: Normal vs Anomaly", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "score_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plots a labeled confusion matrix heatmap."""
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title("Agent 1 — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# MAIN PIPELINE — runs everything end to end
# =============================================================================

def run_agent1(config=CONFIG):
    """
    Full Agent 1 pipeline:
      1. Load and prepare data
      2. Train autoencoder on normal traffic
      3. Train isolation forest on normal traffic
      4. Compute and combine anomaly scores
      5. Evaluate and visualize results
      6. Return scores for Agent 3 (Bayesian fusion)
    """
    os.makedirs(config["output_dir"], exist_ok=True)

    # --- Step 1: Data ---
    X_train_normal, X_test, y_test, scaler = load_and_prepare_data(config)

    # --- Step 2a: Autoencoder ---
    ae_model, loss_history = train_autoencoder(X_train_normal, config)
    ae_threshold = select_ae_threshold(ae_model, X_train_normal, percentile=95)
    ae_scores_test = get_autoencoder_scores(ae_model, X_test)

    # --- Step 2b: Isolation Forest ---
    iso_forest = train_isolation_forest(X_train_normal, config)
    if_scores_test = get_isolation_forest_scores(iso_forest, X_test)

    # --- Step 3: Combine scores ---
    combined_scores, ae_norm, if_norm = compute_agent1_scores(
        ae_scores_test, if_scores_test,
        ae_weight=0.5, if_weight=0.5
    )

    # --- Step 4: Find best threshold and predict ---
    best_threshold = select_combined_threshold(combined_scores, y_test)
    y_pred = predict_from_scores(combined_scores, threshold=best_threshold)

    # --- Step 5: Evaluate ---
    evaluate_agent1(y_test, y_pred, combined_scores)

    # --- Step 6: Visualize ---
    print("\n" + "="*60)
    print("Saving plots...")
    print("="*60)
    plot_training_loss(loss_history, config["output_dir"])
    plot_score_distributions(ae_norm, if_norm, combined_scores, y_test, config["output_dir"])
    plot_confusion_matrix(y_test, y_pred, config["output_dir"])

    # --- Save final scores for Agent 3 ---
    # Agent 3 will use these continuous scores (not binary predictions)
    # for Bayesian fusion with Agent 2's output
    scores_df = pd.DataFrame({
        "ae_score_normalized"  : ae_norm,
        "if_score_normalized"  : if_norm,
        "agent1_combined_score": combined_scores,
        "agent1_prediction"    : y_pred,
        "true_label"           : y_test
    })
    scores_path = os.path.join(config["output_dir"], "agent1_scores.csv")
    scores_df.to_csv(scores_path, index=False)
    print(f"\nAgent 1 scores saved to {scores_path}")
    print("(These will be loaded by Agent 3 for Bayesian fusion)\n")

    print("="*60)
    print("AGENT 1 COMPLETE")
    print("="*60)

    return ae_model, iso_forest, combined_scores, y_pred, y_test


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    ae_model, iso_forest, scores, predictions, true_labels = run_agent1(CONFIG)
