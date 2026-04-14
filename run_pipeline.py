"""
MA-NIDS Pipeline Runner
Runs the full Multi-Agent Network Intrusion Detection System in order:
  Step 1 - Preprocessing           : cleans data, balances classes, saves indices
  Step 2 - Agent 1 (train)         : trains autoencoder on normal traffic
  Step 3 - Agent 1 (save scores)   : computes training scores for Agent 3
  Step 4 - Agent 2 (train)         : trains Random Forest + XGBoost classifiers
  Step 5 - Agent 2 (threshold)     : tunes Infilteration threshold to reduce FAR
  Step 6 - Agent 3 (Bayesian)      : probabilistic fusion using Bayes' Theorem
  Step 7 - Agent 3 (Stacking)      : meta-learning fusion via Logistic Regression

Usage:
    python run_pipeline.py              # run full pipeline
    python run_pipeline.py --from 2     # skip preprocessing
    python run_pipeline.py --from 4     # skip to Agent 2 training
    python run_pipeline.py --from 6     # skip to Agent 3 (Bayesian fusion) only
    python run_pipeline.py --from 7     # skip to Agent 3 (Stacking) only
"""

import argparse
import os
import subprocess
import sys
import time

# Step definitions

STEPS = [
    {
        "number": 1,
        "name": "Preprocessing",
        "script": "data/preprocess.py",
        "desc": "Cleans data, balances classes, saves shared train/test indices"
    },
    {
        "number": 2,
        "name": "Agent 1 - Anomaly Detector",
        "script": "agent1/agent1.py",
        "desc": "Trains autoencoder + isolation forest on normal traffic only"
    },
    {
        "number": 3,
        "name": "Agent 1 - Save Training Scores",
        "script": "agent1/save_train_scores.py",
        "desc": "Computes Agent 1 scores on training set for Agent 3 likelihood estimation"
    },
    {
        "number": 4,
        "name": "Agent 2 - Supervised Classifier",
        "script": "agent2/train.py",
        "desc": "Trains Random Forest + XGBoost on labeled traffic data"
    },
    {
        "number": 5,
        "name": "Agent 2 - Threshold Tuning",
        "script": "agent2/threshold_tuning.py",
        "desc": "Tunes Infilteration confidence threshold to reduce false alarms"
    },
    {
        "number": 6,
        "name": "Agent 3 - Bayesian Fusion",
        "script": "agent3/fusion.py",
        "desc": "Combines Agent 1 and Agent 2 outputs into final threat score"
    },
    {
        "number": 7,
        "name": "Agent 3 - Stacking",
        "script": "agent3/stacking.py",
        "desc": "Meta-learning fusion - Logistic Regression learns to combine agents"
    },
]


# Helper

def run_step(step):
    """
    Runs a single pipeline step as a subprocess.
    Changes into the script's own directory first so relative
    paths like ../data/processed/ resolve correctly.
    Stops the entire pipeline if a step fails.
    """
    print("\n" + "=" * 60)
    print(f"  STEP {step['number']}: {step['name']}")
    print(f"  {step['desc']}")
    print(f"  Running: {step['script']}")
    print("=" * 60 + "\n")

    if not os.path.exists(step["script"]):
        print(f"  ERROR: {step['script']} not found. Skipping.")
        return False

    t_start = time.time()
    script_dir = os.path.dirname(os.path.abspath(step["script"]))
    script_file = os.path.basename(step["script"])

    # Run from the script's own directory so relative paths resolve correctly
    result = subprocess.run(
        [sys.executable, script_file],
        cwd=script_dir  # ← key fix: sets working directory to script's folder
    )

    elapsed = time.time() - t_start

    if result.returncode != 0:
        print(f"\n  FAILED: {step['name']} exited with error code {result.returncode}")
        print(f"  Pipeline stopped. Fix the error above and rerun with --from {step['number']}")
        return False

    print(f"\n  DONE: {step['name']} completed in {elapsed:.1f}s")
    return True


# Main

def main():
    parser = argparse.ArgumentParser(description="MA-NIDS Pipeline Runner")
    parser.add_argument(
        "--from",
        type=int,
        default=1,
        dest="start_from",
        choices=list(range(1, len(STEPS) + 1)),
        help=(
            "Step to start from: 1=preprocess, 2=agent1-train, "
            "3=agent1-save-scores, 4=agent2-train, 5=agent2-threshold, "
            "6=agent3-bayesian, 7=agent3-stacking"
        )
    )
    args = parser.parse_args()

    # Filter steps based on start_from argument
    steps_to_run = [s for s in STEPS if s["number"] >= args.start_from]

    print("\n" + "=" * 60)
    print("  MA-NIDS - Full Pipeline")
    print("=" * 60)
    print(f"  Starting from Step {args.start_from}")
    print(f"  Steps to run: {[s['name'] for s in steps_to_run]}")

    total_start = time.time()

    for step in steps_to_run:
        success = run_step(step)
        if not success:
            sys.exit(1)  # stop pipeline on failure

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Total time: {total_elapsed:.1f}s")
    print("=" * 60)
    print("\nOutputs saved to data/processed/")
    print("  cleaned_data.csv                    - preprocessed dataset")
    print("  label_encoder.pkl                   - class name mapping")
    print("  train_idx.npy / test_idx.npy         - shared split indices")
    print("  agent1_outputs/                      - Agent 1 models and scores")
    print("  agent1_outputs/agent1_train_scores.csv - training scores for Agent 3")
    print("  agent2_Random_Forest.pkl             - Agent 2 trained model")
    print("  agent2_Random_Forest_test_predictions.csv - Agent 2 probabilities")
    print("  agent3_outputs/                      - Agent 3 fusion results")


if __name__ == "__main__":
    main()
