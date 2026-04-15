"""
Generate Figures

Generates three visualizations not produced by the main pipeline:
  1. FAR comparison bar chart across all methods
  2. Agent 1 per-class anomaly score bar chart
  3. Per-class F1 comparison: Agent 2 vs Agent 3 Stacking

Run from the ma-nids/ root directory after the full pipeline has completed.
Output saved to data/processed/paper_figures/
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "data/processed/paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette
BLUE = "#2C7BB6"
ORANGE = "#D7191C"
GREEN = "#1A9641"
PURPLE = "#7B2D8B"
LIGHT_BLUE = "#ABD9E9"
LIGHT_RED = "#FDAE61"
GREY = "#AAAAAA"

# =============================================================================
# FIGURE 1 - FAR Comparison Across All Methods
# =============================================================================
# Headline result - shows the progression of FAR improvements across every approach in the system.

print("Generating Figure 1: FAR comparison...")

methods = [
    "Agent 1\n(Standalone)",
    "Agent 2\n(Random Forest)",
    "Agent 3\n(Bayesian)",
    "Agent 3\n(Stacking)",
    "Agent 2 +\nThreshold"
]

far_values = [46.19, 33.16, 33.18, 30.30, 16.15]

colors = [GREY, BLUE, LIGHT_BLUE, GREEN, ORANGE]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(methods, far_values, color=colors,
               edgecolor='white', height=0.55)

# Annotate bars with exact values
for bar, val in zip(bars, far_values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%", va='center', ha='left',
            fontsize=11, fontweight='bold')

# Vertical line at Agent 2 baseline for reference
ax.axvline(x=33.16, color=BLUE, linestyle='--',
           linewidth=1.2, alpha=0.6, label='Agent 2 baseline (33.16%)')

ax.set_xlabel("False Alarm Rate (%)", fontsize=12)
ax.set_title("False Alarm Rate Comparison Across All Methods\n"
             "Lower is better - % of normal traffic incorrectly flagged as attack",
             fontsize=13, fontweight='bold')
ax.set_xlim(0, 58)
ax.invert_yaxis()  # best result at bottom
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig1_far_comparison.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {path}")

# =============================================================================
# FIGURE 2 - Agent 1 Per-Class Anomaly Score
# =============================================================================
# Shows which attack classes the autoencoder can detect (high score)
# and which blend in with normal traffic (score near Benign baseline).

print("Generating Figure 2: Agent 1 per-class anomaly scores...")

# Mean combined scores from save_train_scores.py output
# Values from final pipeline run
classes = [
    "DoS-GoldenEye",
    "SSH-Bruteforce",
    "DDOS-HOIC",
    "SlowHTTPTest",
    "FTP-BruteForce",
    "Infilteration",
    "DoS-Hulk",
    "Benign"
]

scores = [0.006459, 0.004822, 0.002535, 0.001348,
          0.001345, 0.000866, 0.000830, 0.000520]

# Color: attacks that Agent 1 detects well vs poorly vs Benign baseline
bar_colors = [
    GREEN,  # GoldenEye   - strong signal
    GREEN,  # SSH         - strong signal
    LIGHT_BLUE,  # HOIC        - moderate
    LIGHT_RED,  # SlowHTTP    - weak
    LIGHT_RED,  # FTP         - weak
    ORANGE,  # Infilteration - nearly invisible
    ORANGE,  # Hulk        - nearly invisible
    GREY  # Benign      - baseline
]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(classes, scores, color=bar_colors,
               edgecolor='white', height=0.55)

# Annotate with multiplier vs Benign
benign_score = 0.000520
for bar, val, cls in zip(bars, scores, classes):
    multiplier = val / benign_score
    label = f"{val:.4f}  ({multiplier:.1f}× Benign)" if cls != "Benign" \
        else f"{val:.4f}  (baseline)"
    ax.text(bar.get_width() + 0.00005,
            bar.get_y() + bar.get_height() / 2,
            label, va='center', ha='left', fontsize=9.5)

# Legend
legend_patches = [
    mpatches.Patch(color=GREEN, label='Strong signal - reliably detected'),
    mpatches.Patch(color=LIGHT_BLUE, label='Moderate signal'),
    mpatches.Patch(color=LIGHT_RED, label='Weak signal'),
    mpatches.Patch(color=ORANGE, label='Nearly invisible - mimics normal traffic'),
    mpatches.Patch(color=GREY, label='Benign baseline'),
]
ax.legend(handles=legend_patches, fontsize=9, loc='lower right')

ax.set_xlabel("Mean Agent 1 Combined Anomaly Score", fontsize=12)
ax.set_title("Agent 1 - Per-Class Anomaly Score (Training Set)\n"
             "Higher score = more anomalous vs normal traffic",
             fontsize=13, fontweight='bold')
ax.set_xlim(0, 0.010)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig2_agent1_per_class_scores.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {path}")

# =============================================================================
# FIGURE 3 - Per-Class F1: Agent 2 vs Agent 3 Stacking
# =============================================================================
# Shows which classes improved, which stayed the same, and which
# couldn't be helped by fusion.

print("Generating Figure 3: Per-class F1 comparison...")

class_names = [
    "Benign",
    "DDOS-HOIC",
    "DoS-GoldenEye",
    "DoS-Hulk",
    "DoS-SlowHTTPTest",
    "FTP-BruteForce",
    "Infilteration",
    "SSH-Bruteforce"
]

# From final pipeline output
f1_agent2 = [0.7445, 0.9998, 0.9998, 0.9999,
             0.6353, 0.7466, 0.7919, 0.9999]

f1_stacking = [0.7556, 1.0000, 0.9998, 0.9999,
               0.6353, 0.7466, 0.7910, 0.9999]

x = np.arange(len(class_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width / 2, f1_agent2, width,
               label='Agent 2 (Random Forest)', color=BLUE,
               edgecolor='white')
bars2 = ax.bar(x + width / 2, f1_stacking, width,
               label='Agent 3 (Stacking)', color=GREEN,
               edgecolor='white')

# Annotate with change where meaningful
for i, (v2, vs) in enumerate(zip(f1_agent2, f1_stacking)):
    diff = vs - v2
    if abs(diff) >= 0.001:
        ax.text(x[i] + width / 2, vs + 0.005,
                f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}",
                ha='center', va='bottom', fontsize=8.5,
                color=GREEN if diff > 0 else ORANGE,
                fontweight='bold')

ax.set_ylabel("F1 Score", fontsize=12)
ax.set_title("Per-Class F1 Score: Agent 2 vs Agent 3 Stacking\n"
             "Stacking improves Benign classification - the source of all false alarms",
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=20, ha='right', fontsize=10)
ax.set_ylim(0.55, 1.05)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Highlight Benign bar - that's the key result
ax.annotate('Key improvement:\nBenign F1 +0.011',
            xy=(x[0] + width / 2, f1_stacking[0]),
            xytext=(x[0] + 1.2, 0.80),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=9, color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='grey'))

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig3_f1_comparison.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {path}")

# =============================================================================
# DONE
# =============================================================================

print("\nAll figures saved to:", OUTPUT_DIR)
print("Files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")
