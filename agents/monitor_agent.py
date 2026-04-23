"""
agents/monitor_agent.py
-----------------------
Agent 1 — Sensor Monitor

WHAT IT DOES:
  Receives one sensor reading from the stream, runs it through
  the Isolation Forest anomaly model, and writes the result
  to AgentState.

WHY IT'S A SEPARATE AGENT:
  In production, this agent runs every 30 seconds on a Cloud Run
  job triggered by GCP Pub/Sub. It is stateless — it reads one
  message, scores it, and publishes the result. Separating it
  from the diagnostic logic means you can scale them independently.
  10 compressors = 10 parallel monitor agents, 1 diagnostic agent.

WHAT IT WRITES TO STATE:
  is_anomaly, anomaly_score, severity, confidence_pct, fault_type
"""

import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agents.state import AgentState
from ml.anomaly_model import CARISAnomalyDetector

MODEL_DIR = "ml/saved_models"

# load model once at module level — not on every call
# in production this is loaded when the Cloud Run container starts
_detector = None


def get_detector() -> CARISAnomalyDetector:
    global _detector
    if _detector is None:
        _detector = CARISAnomalyDetector.load(MODEL_DIR)
    return _detector


def monitor_agent(state: AgentState) -> AgentState:
    """
    Agent 1 node function for LangGraph.

    Reads sensor features from state, scores them with the
    Isolation Forest model, writes anomaly result back to state.
    """
    log_entry = {
        "agent":     "monitor_agent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action":    "anomaly_detection",
    }

    try:
        detector = get_detector()

        # build feature dict from state
        reading = {
            "rms":          state.get("vibration_rms", 0.0),
            "peak":         state.get("peak", 0.0),
            "kurtosis":     state.get("kurtosis", 0.0),
            "crest_factor": state.get("crest_factor", 0.0),
            "std":          state.get("std", 0.0),
        }

        result = detector.predict_single(reading)

        # map severity to fault type label for Agent 2
        fault_type = _estimate_fault_type(reading)

        log_entry["result"] = {
            "is_anomaly":     result["is_anomaly"],
            "severity":       result["severity"],
            "anomaly_score":  result["anomaly_score"],
            "confidence_pct": result["confidence_pct"],
            "fault_type":     fault_type,
        }

        print(f"[Agent 1] {state.get('equipment_id')} | "
              f"anomaly={result['is_anomaly']} | "
              f"severity={result['severity']} | "
              f"kurtosis={reading['kurtosis']:.2f} | "
              f"fault_type={fault_type}")

        return {
            **state,
            "is_anomaly":     result["is_anomaly"],
            "anomaly_score":  result["anomaly_score"],
            "severity":       result["severity"],
            "confidence_pct": result["confidence_pct"],
            "fault_type":     fault_type,
            "cycle_count":    state.get("cycle_count", 0) + 1,
            "agent_log":      state.get("agent_log", []) + [log_entry],
        }

    except Exception as e:
        log_entry["error"] = str(e)
        print(f"[Agent 1] ERROR: {e}")
        return {
            **state,
            "is_anomaly":  False,
            "severity":    "normal",
            "fault_type":  "unknown",
            "error":       str(e),
            "agent_log":   state.get("agent_log", []) + [log_entry],
        }


def _estimate_fault_type(reading: dict) -> str:
    """
    Heuristic fault type estimation from feature values.

    In production, you would train a multi-class classifier.
    For CARIS, these thresholds come from the CWRU dataset
    and CPChem maintenance manual fault signatures.

    Inner race: very high kurtosis + high crest factor
    Outer race: high kurtosis + elevated RMS
    Ball fault: moderate kurtosis + moderate crest factor
    """
    kurtosis     = reading.get("kurtosis", 0)
    crest_factor = reading.get("crest_factor", 0)
    rms          = reading.get("rms", 0)

    if kurtosis > 10 and crest_factor > 6:
        return "inner_race"
    elif kurtosis > 7 and rms > 3.5:
        return "outer_race"
    elif kurtosis > 5:
        return "ball"
    else:
        return "unknown"