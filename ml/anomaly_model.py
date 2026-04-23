"""
ml/anomaly_model.py
-------------------
Isolation Forest anomaly detector for CPChem Cedar Bayou rotating equipment.

WHY ISOLATION FOREST:
  - Unsupervised: trains ONLY on normal data. No labelled failures needed.
  - At CPChem, we have years of normal operation data but very few
    documented failures. Supervised models can't work here.
  - Fast inference: scores one window in microseconds (critical for
    real-time 30-second monitoring loops in Agent 1)
  - Explainable: contamination parameter has clear physical meaning
  - Industry standard for anomaly detection in OT/industrial settings

HOW IT WORKS:
  Isolation Forest randomly partitions the feature space using decision
  trees. Normal points require many splits to isolate (they cluster with
  others). Anomalous points are isolated quickly (they're far from the
  cluster). The anomaly score = average path length across all trees.
  Short path = anomaly. Long path = normal.

FEATURE MEANINGS FOR INTERVIEW:
  rms           -> overall vibration energy. Rising RMS = increasing wear
  peak          -> maximum spike. Sudden jump = impact fault
  kurtosis      -> impulsiveness. >10 strongly suggests bearing fault
  crest_factor  -> peak/rms ratio. High = localized fault (not global wear)
  std           -> signal spread. Increases as fault severity grows
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime


FEATURE_COLS = ["rms", "peak", "kurtosis", "crest_factor", "std"]
MODEL_DIR    = "ml/saved_models"
MODEL_PATH   = os.path.join(MODEL_DIR, "isolation_forest.joblib")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


class CARISAnomalyDetector:
    """
    Wraps Isolation Forest + StandardScaler into one object.

    Why scale features?
      kurtosis can range 3-50, rms might be 0.001-0.5 depending on units.
      Without scaling, high-magnitude features dominate the tree splits.
      StandardScaler brings all features to mean=0, std=1.

    Usage:
      detector = CARISAnomalyDetector(contamination=0.05)
      detector.fit(normal_df)
      result = detector.predict(new_reading)
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100,
                 random_state: int = 42):
        """
        Parameters
        ----------
        contamination : float
            Expected fraction of anomalies in training data.
            0.05 = we expect 5% of even 'normal' windows to be borderline.
            Lower = stricter (more false negatives).
            Higher = looser (more false positives).
            For CPChem: 0.05 is conservative — we'd rather catch more faults.

        n_estimators : int
            Number of isolation trees. More = more stable scores but slower.
            100 is the standard starting point.
        """
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.random_state  = random_state
        self.model         = None
        self.scaler        = None
        self.threshold     = None
        self.is_fitted     = False
        self.train_stats   = {}

    def _validate_features(self, df: pd.DataFrame) -> np.ndarray:
        """Check all required feature columns are present and return numpy array."""
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns: {missing}\n"
                f"Required: {FEATURE_COLS}\n"
                f"Got: {list(df.columns)}"
            )
        return df[FEATURE_COLS].values.astype(np.float64)

    def fit(self, df: pd.DataFrame) -> "CARISAnomalyDetector":
        """
        Train on NORMAL data only.

        Parameters
        ----------
        df : DataFrame containing only normal (healthy) bearing windows.
             Must have columns: rms, peak, kurtosis, crest_factor, std

        Returns self for method chaining.
        """
        print(f"Training on {len(df)} normal windows...")
        print(f"  contamination={self.contamination}, n_estimators={self.n_estimators}")

        X = self._validate_features(df)

        # scale features to zero mean, unit variance
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # train isolation forest
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_scaled)

        # compute threshold from training scores
        # scores are negative: more negative = more anomalous
        train_scores = self.model.score_samples(X_scaled)
        self.threshold = float(np.percentile(train_scores, self.contamination * 100))

        # store training statistics for MLflow logging
        self.train_stats = {
            "n_train_samples":  len(df),
            "contamination":    self.contamination,
            "n_estimators":     self.n_estimators,
            "threshold":        round(self.threshold, 6),
            "feature_means":    {c: round(float(df[c].mean()), 6) for c in FEATURE_COLS},
            "feature_stds":     {c: round(float(df[c].std()),  6) for c in FEATURE_COLS},
            "trained_at":       datetime.utcnow().isoformat(),
        }

        self.is_fitted = True
        print(f"  Threshold set to: {self.threshold:.6f}")
        print("Training complete.")
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return raw anomaly scores for each row.
        More negative = more anomalous.
        Scores below self.threshold = anomaly.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before score()")
        X = self._validate_features(df)
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomaly for each row.

        Returns DataFrame with added columns:
          anomaly_score    : raw Isolation Forest score (lower = more anomalous)
          is_anomaly       : True if score below threshold
          severity         : 'normal' | 'warning' | 'critical'
          confidence_pct   : how far below threshold (0-100)

        Severity mapping for CPChem:
          normal   -> no action needed
          warning  -> schedule inspection within 72 hours (P3 work order)
          critical -> immediate action required (P1/P2 work order)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")

        scores = self.score(df)
        result = df.copy()
        result["anomaly_score"] = scores

        # binary flag
        result["is_anomaly"] = scores < self.threshold

        # severity based on how far below threshold
        def get_severity(score):
            if score >= self.threshold:
                return "normal"
            gap = self.threshold - score
            if gap < 0.05:
                return "warning"
            return "critical"

        result["severity"] = [get_severity(s) for s in scores]

        # confidence: percentage distance below threshold (capped at 100)
        result["confidence_pct"] = np.clip(
            ((self.threshold - scores) / abs(self.threshold + 1e-9)) * 100, 0, 100
        ).round(1)

        return result

    def predict_single(self, reading: dict) -> dict:
        """
        Predict anomaly for one sensor reading dict.
        This is what Agent 1 calls every 30 seconds.

        Parameters
        ----------
        reading : dict with keys matching FEATURE_COLS
                  e.g. {"rms": 0.12, "peak": 0.45, "kurtosis": 3.2,
                        "crest_factor": 3.75, "std": 0.08}

        Returns
        -------
        dict with prediction results ready to publish to Pub/Sub
        """
        df = pd.DataFrame([reading])
        result = self.predict(df)
        row = result.iloc[0]

        return {
            "is_anomaly":     bool(row["is_anomaly"]),
            "severity":       row["severity"],
            "anomaly_score":  round(float(row["anomaly_score"]), 6),
            "confidence_pct": float(row["confidence_pct"]),
            "threshold":      round(float(self.threshold), 6),
            "features_used":  FEATURE_COLS,
        }

    def evaluate(self, test_df: pd.DataFrame,
                 true_label_col: str = "fault_type") -> dict:
        """
        Evaluate model on labelled test data.

        Parameters
        ----------
        test_df        : DataFrame with feature columns + fault_type column
        true_label_col : column name containing ground truth fault labels

        Returns
        -------
        dict with precision, recall, f1, confusion matrix
        """
        if true_label_col not in test_df.columns:
            raise ValueError(f"Column '{true_label_col}' not in DataFrame")

        result = self.predict(test_df)

        # ground truth: 1 = anomaly (any fault), 0 = normal
        y_true = (test_df[true_label_col] != "normal").astype(int).values
        y_pred = result["is_anomaly"].astype(int).values

        report = classification_report(y_true, y_pred,
                                       target_names=["normal", "anomaly"],
                                       output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "precision_anomaly": round(report["anomaly"]["precision"], 4),
            "recall_anomaly":    round(report["anomaly"]["recall"],    4),
            "f1_anomaly":        round(report["anomaly"]["f1-score"],  4),
            "precision_normal":  round(report["normal"]["precision"],  4),
            "recall_normal":     round(report["normal"]["recall"],     4),
            "accuracy":          round(report["accuracy"],             4),
            "true_negatives":    int(cm[0][0]),
            "false_positives":   int(cm[0][1]),
            "false_negatives":   int(cm[1][0]),
            "true_positives":    int(cm[1][1]),
            "n_test_samples":    len(test_df),
        }

        print("\n=== Evaluation Results ===")
        print(f"  Accuracy:          {metrics['accuracy']:.1%}")
        print(f"  Anomaly Precision: {metrics['precision_anomaly']:.1%}")
        print(f"  Anomaly Recall:    {metrics['recall_anomaly']:.1%}")
        print(f"  Anomaly F1:        {metrics['f1_anomaly']:.1%}")
        print(f"  True Positives:    {metrics['true_positives']}")
        print(f"  False Positives:   {metrics['false_positives']}")
        print(f"  False Negatives:   {metrics['false_negatives']}")

        return metrics

    def save(self, model_dir: str = MODEL_DIR) -> None:
        """Save model, scaler, and metadata to disk."""
        if not self.is_fitted:
            raise RuntimeError("Train the model before saving")

        os.makedirs(model_dir, exist_ok=True)
        model_path    = os.path.join(model_dir, "isolation_forest.joblib")
        scaler_path   = os.path.join(model_dir, "scaler.joblib")
        metadata_path = os.path.join(model_dir, "model_metadata.json")

        joblib.dump(self.model,  model_path)
        joblib.dump(self.scaler, scaler_path)

        with open(metadata_path, "w") as f:
            json.dump(self.train_stats, f, indent=2)

        print(f"Model saved to {model_dir}/")

    @classmethod
    def load(cls, model_dir: str = MODEL_DIR) -> "CARISAnomalyDetector":
        """Load a saved model from disk."""
        model_path    = os.path.join(model_dir, "isolation_forest.joblib")
        scaler_path   = os.path.join(model_dir, "scaler.joblib")
        metadata_path = os.path.join(model_dir, "model_metadata.json")

        for path in [model_path, scaler_path, metadata_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    "Run python -m ml.train first"
                )

        with open(metadata_path) as f:
            metadata = json.load(f)

        detector           = cls(contamination=metadata["contamination"],
                                 n_estimators=metadata["n_estimators"])
        detector.model     = joblib.load(model_path)
        detector.scaler    = joblib.load(scaler_path)
        detector.threshold = metadata["threshold"]
        detector.train_stats = metadata
        detector.is_fitted = True

        print(f"Model loaded from {model_dir}/")
        print(f"  Trained on {metadata['n_train_samples']} samples")
        print(f"  Threshold: {metadata['threshold']}")
        return detector