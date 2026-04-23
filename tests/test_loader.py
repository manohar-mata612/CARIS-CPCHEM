"""
tests/test_loader.py
--------------------
Tests for the CWRU data loader.
Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.loader import mat_to_dataframe, load_all_mat_files, CWRU_FILE_MAP


class TestFeatureExtraction:
    """Test that feature extraction math is correct."""

    def test_rms_of_constant_signal(self, tmp_path):
        """RMS of all-ones signal = 1.0"""
        import scipy.io
        mat_path = tmp_path / "97.mat"
        # create a fake .mat file with 2048 samples of constant 1.0
        signal = np.ones(2048, dtype=np.float64).reshape(-1, 1)
        scipy.io.savemat(str(mat_path), {"X097_DE_time": signal, "X097RPM": np.array([[1797.0]])})

        df = mat_to_dataframe(str(mat_path), window_size=1024)
        assert len(df) == 2, "Expected 2 windows from 2048 samples with window_size=1024"
        assert abs(df["rms"].iloc[0] - 1.0) < 1e-4, "RMS of all-ones should be 1.0"

    def test_peak_is_max_absolute(self, tmp_path):
        """Peak should be the max absolute value in the window."""
        import scipy.io
        mat_path = tmp_path / "97.mat"
        signal = np.linspace(-5, 5, 1024).reshape(-1, 1)
        scipy.io.savemat(str(mat_path), {"X097_DE_time": signal, "X097RPM": np.array([[1797.0]])})

        df = mat_to_dataframe(str(mat_path), window_size=1024)
        assert abs(df["peak"].iloc[0] - 5.0) < 0.1

    def test_crest_factor_positive(self, tmp_path):
        """Crest factor must always be >= 1.0 for any real signal."""
        import scipy.io
        mat_path = tmp_path / "97.mat"
        np.random.seed(42)
        signal = np.random.randn(2048).reshape(-1, 1)
        scipy.io.savemat(str(mat_path), {"X097_DE_time": signal, "X097RPM": np.array([[1797.0]])})

        df = mat_to_dataframe(str(mat_path), window_size=1024)
        assert (df["crest_factor"] >= 1.0).all(), "Crest factor must be >= 1.0"

    def test_dataframe_columns(self, tmp_path):
        """DataFrame must have all required columns for Agent 1."""
        import scipy.io
        mat_path = tmp_path / "97.mat"
        signal = np.random.randn(1024).reshape(-1, 1)
        scipy.io.savemat(str(mat_path), {"X097_DE_time": signal, "X097RPM": np.array([[1797.0]])})

        df = mat_to_dataframe(str(mat_path), window_size=1024)
        required = ["rms", "peak", "kurtosis", "crest_factor", "std",
                    "fault_type", "equipment_id", "filename"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_equipment_id_is_cpchem_format(self, tmp_path):
        """Equipment ID must match CPChem tag format CB-XXX-NNN."""
        import scipy.io
        mat_path = tmp_path / "97.mat"
        signal = np.random.randn(1024).reshape(-1, 1)
        scipy.io.savemat(str(mat_path), {"X097_DE_time": signal})

        df = mat_to_dataframe(str(mat_path))
        assert df["equipment_id"].iloc[0] == "CB-CGC-001"

    def test_fault_type_labelled_correctly(self, tmp_path):
        """Files in CWRU_FILE_MAP must get the correct fault label."""
        import scipy.io
        # 130.mat = outer race fault per CWRU_FILE_MAP
        mat_path = tmp_path / "130.mat"
        signal = np.random.randn(1024).reshape(-1, 1)
        scipy.io.savemat(str(mat_path), {"X130_DE_time": signal})

        df = mat_to_dataframe(str(mat_path))
        assert df["fault_type"].iloc[0] == "outer_race"

    def test_no_nan_values_in_features(self, tmp_path):
        """Feature columns must not contain NaN."""
        import scipy.io
        mat_path = tmp_path / "97.mat"
        signal = np.random.randn(4096).reshape(-1, 1)
        scipy.io.savemat(str(mat_path), {"X097_DE_time": signal})

        df = mat_to_dataframe(str(mat_path))
        feature_cols = ["rms", "peak", "kurtosis", "crest_factor", "std"]
        for col in feature_cols:
            assert not df[col].isna().any(), f"NaN found in column: {col}"


class TestFileLoading:
    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            load_all_mat_files("/nonexistent/path/xyz")

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_all_mat_files(str(tmp_path))

    def test_cwru_file_map_completeness(self):
        """CWRU_FILE_MAP should cover normal + 3 fault types × 4 loads = 16 files."""
        assert len(CWRU_FILE_MAP) == 16
        fault_types = {v[0] for v in CWRU_FILE_MAP.values()}
        assert "normal" in fault_types
        assert "inner_race" in fault_types
        assert "ball" in fault_types
        assert "outer_race" in fault_types