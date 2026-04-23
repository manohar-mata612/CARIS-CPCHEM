"""
tests/test_anomaly.py
---------------------
Tests for the CARIS anomaly detector.
Run: python -m pytest tests/test_anomaly.py -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml.anomaly_model import CARISAnomalyDetector, FEATURE_COLS


def make_normal_data(n=200, seed=42):
    """Generate synthetic normal bearing data."""
    np.random.seed(seed)
    return pd.DataFrame({
        "rms":          np.random.normal(0.10, 0.01, n),
        "peak":         np.random.normal(0.35, 0.03, n),
        "kurtosis":     np.random.normal(3.0,  0.3,  n),
        "crest_factor": np.random.normal(3.5,  0.2,  n),
        "std":          np.random.normal(0.08, 0.01, n),
        "fault_type":   ["normal"] * n,
    })


def make_fault_data(n=50, seed=99):
    """Generate synthetic faulty bearing data — high kurtosis, high peak."""
    np.random.seed(seed)
    return pd.DataFrame({
        "rms":          np.random.normal(0.25, 0.05, n),
        "peak":         np.random.normal(1.20, 0.20, n),
        "kurtosis":     np.random.normal(18.0, 4.0,  n),
        "crest_factor": np.random.normal(8.0,  1.5,  n),
        "std":          np.random.normal(0.22, 0.04, n),
        "fault_type":   ["inner_race"] * n,
    })


class TestModelTraining:

    def test_fit_runs_without_error(self):
        detector = CARISAnomalyDetector()
        detector.fit(make_normal_data())
        assert detector.is_fitted

    def test_threshold_is_set_after_fit(self):
        detector = CARISAnomalyDetector()
        detector.fit(make_normal_data())
        assert detector.threshold is not None
        assert isinstance(detector.threshold, float)

    def test_predict_before_fit_raises(self):
        detector = CARISAnomalyDetector()
        with pytest.raises(RuntimeError):
            detector.predict(make_normal_data(10))

    def test_missing_feature_column_raises(self):
        detector = CARISAnomalyDetector()
        detector.fit(make_normal_data())
        bad_df = pd.DataFrame({"rms": [0.1], "peak": [0.3]})
        with pytest.raises(ValueError):
            detector.predict(bad_df)


class TestPredictions:

    def setup_method(self):
        self.detector = CARISAnomalyDetector(contamination=0.05)
        self.detector.fit(make_normal_data(300))

    def test_normal_data_mostly_not_anomaly(self):
        """At least 90% of normal windows should score as normal."""
        result = self.detector.predict(make_normal_data(100, seed=1))
        normal_rate = (~result["is_anomaly"]).mean()
        assert normal_rate >= 0.90, f"Normal rate too low: {normal_rate:.1%}"

    def test_fault_data_mostly_anomaly(self):
        """At least 80% of fault windows should be detected."""
        result = self.detector.predict(make_fault_data(50))
        anomaly_rate = result["is_anomaly"].mean()
        assert anomaly_rate >= 0.80, f"Detection rate too low: {anomaly_rate:.1%}"

    def test_output_columns_present(self):
        result = self.detector.predict(make_normal_data(10))
        for col in ["is_anomaly", "anomaly_score", "severity", "confidence_pct"]:
            assert col in result.columns

    def test_severity_values_valid(self):
        result = self.detector.predict(
            pd.concat([make_normal_data(20), make_fault_data(20)])
        )
        valid = {"normal", "warning", "critical"}
        assert set(result["severity"].unique()).issubset(valid)

    def test_confidence_between_0_and_100(self):
        result = self.detector.predict(make_fault_data(20))
        assert (result["confidence_pct"] >= 0).all()
        assert (result["confidence_pct"] <= 100).all()

    def test_predict_single_returns_dict(self):
        reading = {
            "rms": 0.25, "peak": 1.2, "kurtosis": 18.0,
            "crest_factor": 8.0, "std": 0.22
        }
        result = self.detector.predict_single(reading)
        assert isinstance(result, dict)
        assert "is_anomaly" in result
        assert "severity" in result
        assert result["is_anomaly"] is True

    def test_predict_single_normal_not_anomaly(self):
        reading = {
            "rms": 0.10, "peak": 0.35, "kurtosis": 3.0,
            "crest_factor": 3.5, "std": 0.08
        }
        result = self.detector.predict_single(reading)
        assert result["is_anomaly"] is False


class TestSaveLoad:

    def test_save_and_load(self, tmp_path):
        detector = CARISAnomalyDetector(contamination=0.05)
        detector.fit(make_normal_data(200))

        detector.save(str(tmp_path))

        loaded = CARISAnomalyDetector.load(str(tmp_path))
        assert loaded.is_fitted
        assert abs(loaded.threshold - detector.threshold) < 1e-9

    def test_loaded_model_gives_same_predictions(self, tmp_path):
        detector = CARISAnomalyDetector(contamination=0.05)
        detector.fit(make_normal_data(200))
        detector.save(str(tmp_path))

        loaded = CARISAnomalyDetector.load(str(tmp_path))
        test_data = make_fault_data(20)

        orig_scores   = detector.score(test_data)
        loaded_scores = loaded.score(test_data)
        np.testing.assert_array_almost_equal(orig_scores, loaded_scores)


class TestEvaluation:

    def test_evaluate_returns_all_metrics(self):
        detector = CARISAnomalyDetector(contamination=0.05)
        detector.fit(make_normal_data(200))

        test_df = pd.concat([make_normal_data(50), make_fault_data(50)])
        metrics = detector.evaluate(test_df)

        required_keys = [
            "accuracy", "precision_anomaly", "recall_anomaly",
            "f1_anomaly", "true_positives", "false_positives",
            "false_negatives", "true_negatives"
        ]
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_metrics_in_valid_range(self):
        detector = CARISAnomalyDetector(contamination=0.05)
        detector.fit(make_normal_data(200))
        test_df = pd.concat([make_normal_data(50), make_fault_data(50)])
        metrics = detector.evaluate(test_df)

        for key in ["accuracy", "precision_anomaly", "recall_anomaly", "f1_anomaly"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"