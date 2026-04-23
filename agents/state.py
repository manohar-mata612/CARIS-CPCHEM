"""
agents/state.py
---------------
Defines the shared state object that flows through the LangGraph.

WHY STATE MATTERS:
  In LangGraph, every agent reads from and writes to one shared
  AgentState object. This is how agents hand off information
  to each other without making API calls between themselves.

  Think of it like a shared whiteboard:
    - Agent 1 writes the anomaly it detected
    - Agent 2 reads that anomaly, writes its diagnosis
    - Agent 3 reads that diagnosis, writes the work order
    - Orchestrator reads everything, decides what to do next

  All of this happens through one Python TypedDict.
"""

from typing import TypedDict, Optional


class AgentState(TypedDict):
    """
    Shared state flowing through all 4 CARIS agents.
    Every field is Optional — agents only fill in what they know.
    """

    # --- Input: raw sensor reading ---
    equipment_id:       str
    vibration_rms:      float
    kurtosis:           float
    crest_factor:       float
    std:                float
    peak:               float
    timestamp:          str

    # --- Agent 1 output: anomaly detection result ---
    is_anomaly:         bool
    anomaly_score:      float
    severity:           str        # "normal" | "warning" | "critical"
    confidence_pct:     float
    fault_type:         str        # "inner_race" | "ball" | "outer_race"

    # --- Agent 2 output: RAG diagnosis ---
    diagnosis:          str
    root_causes:        list
    recommended_actions: list
    parts_required:     list
    estimated_labor_hours: float
    safety_notes:       str
    source_documents:   list

    # --- Agent 3 output: work order ---
    work_order:         dict

    # --- Orchestrator fields ---
    cycle_count:        int
    human_escalated:    bool
    final_priority:     str        # "P1" | "P2" | "P3" | "normal"
    agent_log:          list       # audit trail of every decision
    error:              Optional[str]