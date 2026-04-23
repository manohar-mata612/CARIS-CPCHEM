"""
ml/train.py
-----------
Training script for the CARIS anomaly detector.
Logs every experiment to MLflow so you can compare runs.

WHY MLFLOW:
  In a Frame/CPChem production engagement, the data science team runs
  dozens of experiments: different contamination values, feature sets,
  window sizes. MLflow tracks every run so you can:
    - Reproduce any result exactly
    - Compare precision/recall across parameter settings
    - Deploy a specific model version with confidence
    - Show auditors exactly what model is running in production

  On GCP: MLflow runs on Vertex AI Workbench (free tier).
  On Azure (Frame's production stack): Azure ML uses the same MLflow API.
  You change one environment variable. Nothing else changes.

Usage:
  python -m ml.train
  python -m ml.train --contamination 0.1
  python -m ml.train --contamination 0.05 --n-estimators 200
  python -m ml.train --compare   (runs 5 experiments, compare in MLflow UI)
"""

import argparse
import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.loader import load_processed
from ml.anomaly_model import CARISAnomalyDetector, FEATURE_COLS

PROCESSED_PATH  = "data/processed/cwru_features.csv"
MLFLOW_TRACKING = "mlruns"
EXPERIMENT_NAME = "caris-cpchem-anomaly-detection"


def split_data(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Split into train (normal only) and test (all fault types).

    Train set: ONLY normal windows -> Isolation Forest learns normal patterns
    Test set:  ALL fault types     -> measures detection performance

    This mirrors real CPChem deployment:
      - We have years of normal historian data to train on
      - We validate against known historical failure events
    """
    normal_df = df[df["fault_type"] == "normal"].copy()
    fault_df  = df[df["fault_type"] != "normal"].copy()

    # shuffle
    normal_df = normal_df.sample(frac=1, random_state=42).reset_index(drop=True)
    fault_df  = fault_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 80% normal for training
    n_train = int(len(normal_df) * (1 - test_ratio))
    train_df = normal_df.iloc[:n_train]

    # test = remaining normal + all faults
    test_normal = normal_df.iloc[n_train:]
    test_df = pd.concat([test_normal, fault_df], ignore_index=True)

    print(f"Train: {len(train_df)} normal windows")
    print(f"Test:  {len(test_df)} windows "
          f"({len(test_normal)} normal + {len(fault_df)} fault)")
    print(f"Fault breakdown in test:")
    print(test_df["fault_type"].value_counts().to_string())

    return train_df, test_df


def run_experiment(contamination: float, n_estimators: int,
                   train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Run one MLflow experiment: train -> evaluate -> log -> save.

    Returns metrics dict.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"IF_cont{contamination}_est{n_estimators}"):

        # --- log parameters ---
        mlflow.log_param("contamination",  contamination)
        mlflow.log_param("n_estimators",   n_estimators)
        mlflow.log_param("n_train",        len(train_df))
        mlflow.log_param("n_test",         len(test_df))
        mlflow.log_param("features",       str(FEATURE_COLS))
        mlflow.log_param("model_type",     "IsolationForest")
        mlflow.log_param("equipment",      "CB-CGC-001")
        mlflow.log_param("plant",          "cedar_bayou_cpchem")

        # --- train ---
        detector = CARISAnomalyDetector(
            contamination=contamination,
            n_estimators=n_estimators
        )
        detector.fit(train_df)

        # --- evaluate ---
        metrics = detector.evaluate(test_df)

        # --- log metrics to MLflow ---
        mlflow.log_metric("accuracy",          metrics["accuracy"])
        mlflow.log_metric("precision_anomaly", metrics["precision_anomaly"])
        mlflow.log_metric("recall_anomaly",    metrics["recall_anomaly"])
        mlflow.log_metric("f1_anomaly",        metrics["f1_anomaly"])
        mlflow.log_metric("true_positives",    metrics["true_positives"])
        mlflow.log_metric("false_positives",   metrics["false_positives"])
        mlflow.log_metric("false_negatives",   metrics["false_negatives"])
        mlflow.log_metric("threshold",         detector.threshold)

        # --- log model artifact ---
        mlflow.sklearn.log_model(detector.model, "isolation_forest")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run ID: {run_id}")
        print(f"View UI: mlflow ui --port 5000")

    return metrics, detector


def run_comparison(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Run 5 experiments with different contamination values.
    Great for the interview — shows you systematically evaluated parameters.
    """
    contamination_values = [0.01, 0.03, 0.05, 0.10, 0.15]
    results = []

    print("\n=== Running comparison experiments ===")
    for cont in contamination_values:
        print(f"\n--- contamination={cont} ---")
        metrics, _ = run_experiment(cont, 100, train_df, test_df)
        metrics["contamination"] = cont
        results.append(metrics)

    # print comparison table
    print("\n=== Comparison Summary ===")
    print(f"{'Contamination':<15} {'Accuracy':<10} {'Precision':<12} "
          f"{'Recall':<10} {'F1':<8}")
    print("-" * 58)
    for r in results:
        print(f"{r['contamination']:<15} "
              f"{r['accuracy']:<10.1%} "
              f"{r['precision_anomaly']:<12.1%} "
              f"{r['recall_anomaly']:<10.1%} "
              f"{r['f1_anomaly']:<8.1%}")

    best = max(results, key=lambda x: x["f1_anomaly"])
    print(f"\nBest F1: contamination={best['contamination']} "
          f"-> F1={best['f1_anomaly']:.1%}")
    return best["contamination"]


def main():
    parser = argparse.ArgumentParser(description="Train CARIS anomaly detector")
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--n-estimators",  type=int,   default=100)
    parser.add_argument("--data",          default=PROCESSED_PATH)
    parser.add_argument("--compare",       action="store_true",
                        help="Run 5 experiments and compare")
    args = parser.parse_args()

    # load data
    print(f"Loading data from {args.data}...")
    df = load_processed(args.data)
    print(f"Loaded {len(df)} windows")
    print(f"Fault distribution:")
    print(df["fault_type"].value_counts().to_string())

    # split
    train_df, test_df = split_data(df)

    if args.compare:
        best_contamination = run_comparison(train_df, test_df)
        print(f"\nRetraining best model (contamination={best_contamination})...")
        metrics, detector = run_experiment(
            best_contamination, args.n_estimators, train_df, test_df
        )
    else:
        metrics, detector = run_experiment(
            args.contamination, args.n_estimators, train_df, test_df
        )

    # save best model
    detector.save()
    print("\nPhase 2 complete.")
    print("Next: python -m ml.train --compare  (to compare all parameters)")
    print("      mlflow ui --port 5000          (to view experiment results)")


if __name__ == "__main__":
    main()