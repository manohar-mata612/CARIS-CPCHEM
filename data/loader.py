"""
data/loader.py
--------------
Loads CWRU .mat files (Kaggle naming convention) and converts to DataFrame.

Kaggle CWRU file naming convention:
  Normal_0.mat  -> 0 HP load, normal baseline
  Normal_1.mat  -> 1 HP load, normal baseline
  IR007_0.mat   -> inner race fault 0.007", 0 HP
  B007_0.mat    -> ball fault 0.007", 0 HP
  OR0076_0.mat  -> outer race fault 0.007" @ 6 o'clock, 0 HP

Suffix _0 _1 _2 _3 = motor load 0 1 2 3 HP
"""

import os
import re
import scipy.io
import pandas as pd
import numpy as np


# Map filename prefix -> (fault_type, fault_location)
FAULT_PREFIX_MAP = {
    "Normal": ("normal",     "none"),
    "IR007":  ("inner_race", "drive_end"),
    "IR014":  ("inner_race", "drive_end"),
    "IR021":  ("inner_race", "drive_end"),
    "B007":   ("ball",       "drive_end"),
    "B014":   ("ball",       "drive_end"),
    "B021":   ("ball",       "drive_end"),
    "OR0076": ("outer_race", "drive_end"),
    "OR0146": ("outer_race", "drive_end"),
    "OR0216": ("outer_race", "drive_end"),
    "OR007":  ("outer_race", "drive_end"),
    "OR014":  ("outer_race", "drive_end"),
    "OR021":  ("outer_race", "drive_end"),
}

# Map filename prefix -> fault diameter in inches
FAULT_DIAMETER_MAP = {
    "Normal": 0.000,
    "IR007":  0.007, "IR014": 0.014, "IR021": 0.021,
    "B007":   0.007, "B014":  0.014, "B021":  0.021,
    "OR0076": 0.007, "OR0146": 0.014, "OR0216": 0.021,
    "OR007":  0.007, "OR014":  0.014, "OR021":  0.021,
}


def parse_filename(filename):
    """
    Parse Kaggle CWRU filename into fault metadata.
    e.g. 'IR007_2.mat' -> fault_type='inner_race', load_hp=2, diameter=0.007
    """
    name = os.path.splitext(filename)[0]  # strip .mat
    parts = name.rsplit("_", 1)           # split on last underscore

    prefix = parts[0]
    load_hp = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

    if prefix in FAULT_PREFIX_MAP:
        fault_type, fault_location = FAULT_PREFIX_MAP[prefix]
        fault_diameter = FAULT_DIAMETER_MAP.get(prefix, 0.0)
    else:
        fault_type, fault_location, fault_diameter = "unknown", "unknown", 0.0

    return fault_type, fault_location, fault_diameter, load_hp


def extract_drive_end_vibration(mat_data):
    """Extract drive-end vibration array from .mat file regardless of key name."""
    for key in mat_data.keys():
        if "DE_time" in key:
            return mat_data[key].flatten()
    # fallback: first non-private numeric array
    for key in mat_data.keys():
        if not key.startswith("__"):
            arr = mat_data[key].flatten()
            if len(arr) > 100:
                return arr
    raise KeyError(f"No vibration data found. Keys: {list(mat_data.keys())}")


def extract_rpm(mat_data):
    """Extract motor RPM from .mat file."""
    for key in mat_data.keys():
        if "RPM" in key.upper():
            return float(np.array(mat_data[key]).flatten()[0])
    return 1797.0


def mat_to_dataframe(filepath, window_size=1024):
    """
    Convert one .mat file to a DataFrame of feature windows.

    Each row = one window with 5 statistical features:
      rms, peak, kurtosis, crest_factor, std

    These 5 features feed the Isolation Forest in Phase 2.
    kurtosis is the key fault indicator:
      healthy bearing -> kurtosis ~3
      faulty bearing  -> kurtosis 10-50+
    """
    filename = os.path.basename(filepath)
    fault_type, fault_location, fault_diameter, load_hp = parse_filename(filename)

    mat_data = scipy.io.loadmat(filepath)
    vibration = extract_drive_end_vibration(mat_data)
    rpm = extract_rpm(mat_data)

    rows = []
    n_windows = len(vibration) // window_size

    for i in range(n_windows):
        window = vibration[i * window_size: (i + 1) * window_size]

        rms          = float(np.sqrt(np.mean(window ** 2)))
        peak         = float(np.max(np.abs(window)))
        kurtosis     = float(pd.Series(window).kurt())
        crest_factor = float(peak / rms) if rms > 0 else 0.0
        std          = float(np.std(window))

        rows.append({
            "window_id":      i,
            "rms":            round(rms, 6),
            "peak":           round(peak, 6),
            "kurtosis":       round(kurtosis, 6),
            "crest_factor":   round(crest_factor, 6),
            "std":            round(std, 6),
            "rpm":            rpm,
            "fault_type":     fault_type,
            "fault_location": fault_location,
            "fault_diameter": fault_diameter,
            "load_hp":        load_hp,
            "equipment_id":   "CB-CGC-001",
            "filename":       filename,
        })

    return pd.DataFrame(rows)


def load_all_mat_files(data_dir, window_size=1024):
    """Load every .mat file in a directory into one combined DataFrame."""
    mat_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]

    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found in {data_dir}\n"
            "Download from Kaggle: cwru-case-western-reserve-university-dataset"
        )

    print(f"Found {len(mat_files)} .mat files in {data_dir}")
    all_dfs = []

    for filename in sorted(mat_files):
        filepath = os.path.join(data_dir, filename)
        try:
            df = mat_to_dataframe(filepath, window_size)
            all_dfs.append(df)
            print(f"  {filename}: {len(df)} windows | "
                  f"fault={df['fault_type'].iloc[0]} | "
                  f"load={df['load_hp'].iloc[0]}HP")
        except Exception as e:
            print(f"  SKIP {filename}: {e}")

    combined = pd.concat(all_dfs, ignore_index=True)

    print(f"\nTotal windows loaded: {len(combined)}")
    print("Fault distribution:")
    print(combined["fault_type"].value_counts().to_string())
    return combined


def save_processed(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path}")


def load_processed(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Processed file not found: {filepath}\n"
            "Run: python -m data.loader data/raw/cwru"
        )
    return pd.read_csv(filepath)


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/cwru"
    df = load_all_mat_files(data_dir)
    save_processed(df, "data/processed/cwru_features.csv")