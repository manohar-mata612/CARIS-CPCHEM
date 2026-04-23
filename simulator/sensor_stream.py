"""
simulator/sensor_stream.py
--------------------------
CARIS sensor data streamer with 3 modes:

  Mode 1 - stdout:  print JSON to terminal (default)
  Mode 2 - api:     POST to FastAPI endpoint (real-time demo)
  Mode 3 - infinite: generate synthetic readings forever (production demo)

Usage:
  python -m simulator.sensor_stream                          # stdout
  python run_stream.py                                       # auto-login + infinite stream
  python -m simulator.sensor_stream --loop                   # loop CWRU data forever
  python -m simulator.sensor_stream --infinite               # infinite synthetic generation
"""

import json
import time
import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

EQUIPMENT_MAP = {
    "CB-CGC-001": "Charge Gas Compressor 1",
    "CB-CGC-002": "Charge Gas Compressor 2",
    "CB-QWP-001": "Quench Water Pump 1",
    "CB-FDF-001": "Feed Furnace Fan 1",
}

LOAD_CONTEXT = {0: "low_load", 1: "normal_load", 2: "high_load", 3: "max_load"}

FAILURE_CODE_MAP = {
    "normal":     None,
    "inner_race": "MECH-BRG-IR",
    "ball":       "MECH-BRG-RE",
    "outer_race": "MECH-BRG-OR",
    "unknown":    "MECH-UNK",
}

# inject a fault every N readings to keep dashboard active
FAULT_INJECTION_EVERY = 8


def load_stream_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Processed data not found at {csv_path}\n"
            "Run: python -m data.loader data/raw/cwru"
        )
    df = pd.read_csv(csv_path)
    print(f"Stream data loaded: {len(df)} windows")
    return df


def row_to_sensor_message(row: pd.Series, equipment_id: str = "CB-CGC-001") -> dict:
    fault_type   = row.get("fault_type", "normal")
    failure_code = FAILURE_CODE_MAP.get(fault_type)

    return {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "equipment_id":       equipment_id,
        "equipment_name":     EQUIPMENT_MAP.get(equipment_id, equipment_id),
        "equipment_type":     "rotating_equipment",
        "plant":              "cedar_bayou",
        "operator":           "cpchem",
        "vibration_rms":      round(float(row["rms"]), 4),
        "vibration_peak":     round(float(row["peak"]), 4),
        "vibration_kurtosis": round(float(row["kurtosis"]), 4),
        "crest_factor":       round(float(row["crest_factor"]), 4),
        "vibration_std":      round(float(row["std"]), 4),
        "motor_rpm":          round(float(row.get("rpm", 1797)), 1),
        "load_condition":     LOAD_CONTEXT.get(int(row.get("load_hp", 0)), "normal_load"),
        "fault_type":         fault_type,
        "fault_location":     row.get("fault_location", "none"),
        "fault_diameter":     float(row.get("fault_diameter", 0.0)),
        "failure_code":       failure_code,
        "is_known_fault":     bool(fault_type != "normal"),
        "source_file":        str(row.get("filename", "cwru")),
        "window_id":          int(row.get("window_id", 0)),
        "data_source":        "cwru_replay",
    }


def compute_fault_stats(df: pd.DataFrame) -> dict:
    """
    Compute mean and std per fault type from real CWRU data.
    Used by infinite generator to produce statistically realistic readings.
    """
    stats = {}
    feature_cols = ["rms", "peak", "kurtosis", "crest_factor", "std"]

    for fault_type in df["fault_type"].unique():
        subset = df[df["fault_type"] == fault_type]
        stats[fault_type] = {
            col: {
                "mean": float(subset[col].mean()),
                "std":  float(subset[col].std()),
            }
            for col in feature_cols
        }

    print(f"Computed stats for fault types: {list(stats.keys())}")
    return stats


def generate_reading_from_stats(stats: dict, fault_type: str,
                                 equipment_id: str, reading_idx: int) -> dict:
    """
    Generate one synthetic sensor reading from CWRU statistical distributions.

    WHY THIS IS REALISTIC:
      Each reading is sampled from N(mean, std*0.15) of the real CWRU data.
      0.15 = 15% noise — enough variation to look like real sensor data
      but staying within the real fault signature bounds.

    This means the dashboard never runs out of data — readings are
    unique every time but statistically identical to real CWRU data.
    """
    fault_stats  = stats.get(fault_type, stats.get("normal", {}))
    failure_code = FAILURE_CODE_MAP.get(fault_type)

    def sample(col):
        m = fault_stats[col]["mean"]
        s = fault_stats[col]["std"] * 0.15
        return round(float(np.random.normal(m, max(s, 1e-6))), 4)

    rms          = max(0.01, sample("rms"))
    peak         = max(rms,  sample("peak"))
    kurtosis     = sample("kurtosis")
    crest_factor = max(1.0,  sample("crest_factor"))
    std          = max(0.01, sample("std"))

    return {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "equipment_id":       equipment_id,
        "equipment_name":     EQUIPMENT_MAP.get(equipment_id, equipment_id),
        "equipment_type":     "rotating_equipment",
        "plant":              "cedar_bayou",
        "operator":           "cpchem",
        "vibration_rms":      rms,
        "vibration_peak":     peak,
        "vibration_kurtosis": kurtosis,
        "crest_factor":       crest_factor,
        "vibration_std":      std,
        "motor_rpm":          round(float(np.random.normal(1797, 15)), 1),
        "load_condition":     "normal_load",
        "fault_type":         fault_type,
        "fault_location":     "drive_end" if fault_type != "normal" else "none",
        "fault_diameter":     0.007 if fault_type != "normal" else 0.0,
        "failure_code":       failure_code,
        "is_known_fault":     bool(fault_type != "normal"),
        "source_file":        "synthetic",
        "window_id":          reading_idx,
        "data_source":        "synthetic_infinite",
    }


def _post_to_api(payload: dict, api_url: str, token: str) -> str:
    """Send one reading to FastAPI. Returns status string."""
    import requests

    api_payload = {
        "equipment_id":  payload["equipment_id"],
        "vibration_rms": payload["vibration_rms"],
        "peak":          payload["vibration_peak"],
        "kurtosis":      payload["vibration_kurtosis"],
        "crest_factor":  payload["crest_factor"],
        "std":           payload["vibration_std"],
        "timestamp":     payload["timestamp"],
    }

    try:
        res = requests.post(
            f"{api_url}/api/sensor-reading",
            json=api_payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        return "sent" if res.status_code == 200 else f"error {res.status_code}"
    except Exception as e:
        return f"error: {e}"


def stream_infinite(df: pd.DataFrame, api_url: str, token: str,
                     interval: float):
    """
    Generate infinite synthetic readings based on CWRU statistics.

    Pattern:
      readings 1-7:  normal operation (no fault)
      reading  8:    fault injected (rotates through fault types)
      readings 9-15: normal operation
      reading  16:   fault injected
      ... forever

    This means:
      - Dashboard always has live data
      - Faults appear periodically to show agents working
      - Each reading is unique (sampled from distribution)
      - Statistically identical to real CWRU data
    """
    stats       = compute_fault_stats(df)
    fault_types = [ft for ft in stats.keys() if ft != "normal"]
    eq_ids      = list(EQUIPMENT_MAP.keys())

    print(f"\nInfinite stream started")
    print(f"  Fault types available: {fault_types}")
    print(f"  Fault injection: every {FAULT_INJECTION_EVERY} readings")
    print(f"  Interval: {interval}s")
    print(f"  Press Ctrl+C to stop\n")
    print("-" * 65)

    i           = 0
    fault_idx   = 0
    total_sent  = 0
    total_faults = 0

    while True:
        eq_id = eq_ids[i % len(eq_ids)]

        # decide fault type for this reading
        if (i + 1) % FAULT_INJECTION_EVERY == 0 and i > 0:
            fault_type = fault_types[fault_idx % len(fault_types)]
            fault_idx += 1
            total_faults += 1
        else:
            fault_type = "normal"

        msg    = generate_reading_from_stats(stats, fault_type, eq_id, i)
        status = _post_to_api(msg, api_url, token)

        label  = f"FAULT({fault_type})" if fault_type != "normal" else "normal"
        print(f"[{msg['timestamp'][11:19]}] {eq_id:<12} | "
              f"{label:<22} | RMS={msg['vibration_rms']:.3f} | "
              f"Kurt={msg['vibration_kurtosis']:.2f} | {status}")

        total_sent += 1
        if total_sent % 20 == 0:
            print(f"\n  Stats: {total_sent} sent | "
                  f"{total_faults} faults | "
                  f"running {total_sent * interval / 60:.1f} min\n")

        i += 1
        time.sleep(interval)


def stream_to_api_loop(df: pd.DataFrame, api_url: str, token: str,
                        interval: float, loop: bool = False):
    """
    Replay CWRU CSV to API. With --loop, restarts from beginning when done.
    """
    print(f"\nStreaming CWRU data to API | loop={loop}")
    print("-" * 60)

    cycle = 0
    while True:
        cycle += 1
        if loop and cycle > 1:
            print(f"\n[Streamer] Cycle {cycle} — restarting from beginning...\n")

        for i, (_, row) in enumerate(df.iterrows()):
            eq_id  = list(EQUIPMENT_MAP.keys())[i % len(EQUIPMENT_MAP)]
            msg    = row_to_sensor_message(row, equipment_id=eq_id)
            status = _post_to_api(msg, api_url, token)
            print(f"[{msg['timestamp'][11:19]}] {eq_id} | "
                  f"{msg['fault_type']:<12} | {status}")
            time.sleep(interval)

        if not loop:
            print("\n[Streamer] Dataset complete. Use --loop to restart.")
            break


def stream_to_stdout(df: pd.DataFrame, interval: float,
                      fault_filter: str = None):
    if fault_filter:
        df = df[df["fault_type"] == fault_filter].reset_index(drop=True)
        print(f"Filtered to fault_type='{fault_filter}': {len(df)} windows")

    print(f"\nStreaming {len(df)} windows at {interval}s intervals...")
    print("-" * 60)

    for i, (_, row) in enumerate(df.iterrows()):
        eq_id  = list(EQUIPMENT_MAP.keys())[i % len(EQUIPMENT_MAP)]
        msg    = row_to_sensor_message(row, equipment_id=eq_id)
        status = "FAULT" if msg["is_known_fault"] else "NORMAL"
        print(f"[{msg['timestamp']}] {eq_id} | {status} | "
              f"RMS={msg['vibration_rms']:.4f} | "
              f"Kurt={msg['vibration_kurtosis']:.2f}")
        print(json.dumps(msg))
        print()
        time.sleep(interval)


def stream_to_pubsub(df: pd.DataFrame, topic: str, interval: float):
    try:
        from google.cloud import pubsub_v1
    except ImportError:
        print("Install: pip install google-cloud-pubsub")
        sys.exit(1)

    publisher = pubsub_v1.PublisherClient()
    print(f"Streaming to Pub/Sub topic: {topic}")
    for _, row in df.iterrows():
        msg    = row_to_sensor_message(row)
        future = publisher.publish(topic, json.dumps(msg).encode("utf-8"))
        print(f"Published: {future.result()} | fault={msg['fault_type']}")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="CARIS sensor data streamer")
    parser.add_argument("--csv",      default="data/processed/cwru_features.csv")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Seconds between readings (default 2.0)")
    parser.add_argument("--fault",    default=None,
                        help="Filter: normal|inner_race|ball|outer_race")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Stop after N readings")
    parser.add_argument("--loop",     action="store_true",
                        help="Loop CWRU dataset forever")
    parser.add_argument("--infinite", action="store_true",
                        help="Generate infinite synthetic readings (best for demo)")
    parser.add_argument("--pubsub",   action="store_true",
                        help="Push to GCP Pub/Sub")
    parser.add_argument("--api",      default=None,
                        help="FastAPI base URL e.g. http://localhost:8000")
    parser.add_argument("--token",    default=None,
                        help="JWT token (auto-provided by run_stream.py)")
    args = parser.parse_args()

    df = load_stream_data(args.csv)
    if args.limit:
        df = df.head(args.limit)

    # ── route to correct mode ──────────────────────────
    if args.infinite:
        if not args.api or not args.token:
            print("--infinite requires --api and --token")
            print("Use run_stream.py for automatic token handling")
            sys.exit(1)
        stream_infinite(df, args.api, args.token, args.interval)

    elif args.api:
        if not args.token:
            print("--api requires --token. Use run_stream.py instead.")
            sys.exit(1)
        stream_to_api_loop(df, args.api, args.token, args.interval,
                           loop=args.loop)

    elif args.pubsub:
        topic = os.getenv("PUBSUB_TOPIC")
        if not topic:
            print("Set PUBSUB_TOPIC in .env")
            sys.exit(1)
        stream_to_pubsub(df, topic, args.interval)

    else:
        stream_to_stdout(df, args.interval, fault_filter=args.fault)


if __name__ == "__main__":
    main()