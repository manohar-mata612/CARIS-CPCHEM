"""
run_stream.py
-------------
One command to start the infinite CARIS sensor stream.

Auto-logs in to FastAPI, gets JWT token, starts infinite
synthetic generation. No copy-pasting needed.

Usage:
  python run_stream.py                    # infinite mode, 15s interval
  python run_stream.py --interval 5       # faster for testing
  python run_stream.py --mode loop        # replay CWRU data in loop
  python run_stream.py --mode cwru        # replay CWRU once then stop

Run this in Terminal 3 alongside:
  Terminal 1: uvicorn api.main:app --reload --port 8000
  Terminal 2: cd frontend && npm run dev
"""

import subprocess
import sys
import argparse
import time

API_URL  = "http://localhost:8000"
USERNAME = "engineer"
PASSWORD = "cpchem2025"


def get_token() -> str:
    """Login to FastAPI and return JWT token."""
    import requests

    print(f"Logging in as '{USERNAME}'...")
    for attempt in range(5):
        try:
            r = requests.post(
                f"{API_URL}/api/login",
                json={"username": USERNAME, "password": PASSWORD},
                timeout=10
            )
            if r.status_code == 200:
                token = r.json()["access_token"]
                print(f"Token obtained successfully")
                return token
            else:
                print(f"Login failed (status {r.status_code}) — retrying...")
        except requests.exceptions.ConnectionError:
            print(f"Cannot reach API at {API_URL} "
                  f"(attempt {attempt+1}/5) — is FastAPI running?")
        time.sleep(3)

    print("\nERROR: Could not connect to FastAPI.")
    print("Make sure Terminal 1 is running:")
    print("  uvicorn api.main:app --reload --port 8000")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="CARIS auto stream launcher")
    parser.add_argument("--interval", type=float, default=60.0,
                        help="Seconds between readings (default 60s)")
    parser.add_argument("--mode", default="infinite",
                        choices=["infinite", "loop", "cwru"],
                        help="infinite=synthetic forever | loop=CWRU loop | cwru=CWRU once")
    args = parser.parse_args()

    print("=" * 55)
    print("  CARIS Stream Launcher")
    print(f"  Mode:     {args.mode}")
    print(f"  Interval: {args.interval}s")
    print(f"  API:      {API_URL}")
    print("=" * 55 + "\n")

    token = get_token()

    # build subprocess command based on mode
    cmd = [
        sys.executable, "-m", "simulator.sensor_stream",
        "--api",      API_URL,
        "--token",    token,
        "--interval", str(args.interval),
    ]

    if args.mode == "infinite":
        cmd.append("--infinite")
        print("Starting infinite synthetic stream...")
        print("Faults injected every 8 readings")
        print("Press Ctrl+C to stop\n")

    elif args.mode == "loop":
        cmd.append("--loop")
        print("Starting CWRU loop stream (restarts when dataset ends)...")
        print("Press Ctrl+C to stop\n")

    elif args.mode == "cwru":
        print("Starting CWRU single-pass stream...")
        print("Will stop when dataset is exhausted\n")

    subprocess.run(cmd)


if __name__ == "__main__":
    main()