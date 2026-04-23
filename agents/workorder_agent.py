"""
agents/workorder_agent.py
-------------------------
Agent 3 — Work Order Generator

WHAT IT DOES:
  Takes the diagnosis from Agent 2 and generates a structured
  SAP PM-format work order JSON. This is the direct output that
  a CPChem maintenance coordinator can copy into their SAP system.

WHY THIS MATTERS FOR THE INTERVIEW:
  The business value of CARIS is not just detecting faults.
  It's compressing the response timeline from 72 hours of human
  coordination to under 10 minutes of automated processing.
  Without this agent, the diagnosis sits in an email.
  With this agent, a work order is generated, prioritized,
  and ready for the SAP system before the engineer even sees
  the alert.

WHAT IT WRITES TO STATE:
  work_order (dict) — complete SAP PM format work order
"""

import os
import sys
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agents.state import AgentState

# SAP PM priority codes used at CPChem Cedar Bayou
SAP_PRIORITY_MAP = {
    "P1": {"code": "1-VH", "description": "Very High — immediate response"},
    "P2": {"code": "2-HI", "description": "High — within 24 hours"},
    "P3": {"code": "3-ME", "description": "Medium — within 72 hours"},
}

# SAP PM failure code -> order type mapping
FAILURE_TO_ORDER_TYPE = {
    "inner_race": "ZM01",  # corrective maintenance
    "outer_race": "ZM01",
    "ball":       "ZM01",
    "unknown":    "ZM02",  # inspection order
}

# work order counter (in production this comes from SAP number range)
_work_order_counter = 1000


def generate_work_order_number() -> str:
    global _work_order_counter
    _work_order_counter += 1
    return f"WO-{_work_order_counter:06d}"


def workorder_agent(state: AgentState) -> AgentState:
    """
    Agent 3 node function for LangGraph.

    Builds a complete SAP PM-style work order from the diagnosis
    in state. Includes equipment tag, failure code, priority,
    parts list, estimated labor, and safety precautions.
    """
    log_entry = {
        "agent":     "workorder_agent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action":    "generate_work_order",
    }

    try:
        priority    = state.get("final_priority", "P2")
        fault_type  = state.get("fault_type", "unknown")
        equipment   = state.get("equipment_id", "CB-CGC-001")
        diagnosis   = state.get("diagnosis", "Bearing fault detected")
        parts       = state.get("parts_required", [])
        labor_hours = state.get("estimated_labor_hours", 8)
        actions     = state.get("recommended_actions", [])
        safety      = state.get("safety_notes", "Follow LOTO procedure SOP-CGC-004")
        sources     = state.get("source_documents", [])

        sap_priority = SAP_PRIORITY_MAP.get(priority, SAP_PRIORITY_MAP["P2"])
        order_type   = FAILURE_TO_ORDER_TYPE.get(fault_type, "ZM02")
        wo_number    = generate_work_order_number()

        # failure code maps fault type to SAP PM code
        failure_code_map = {
            "inner_race": "MECH-BRG-IR",
            "outer_race": "MECH-BRG-OR",
            "ball":       "MECH-BRG-RE",
            "unknown":    "MECH-UNK",
        }
        failure_code = failure_code_map.get(fault_type, "MECH-UNK")

        # build immediate actions list from Agent 2 output
        action_steps = [
            a.get("action", "") for a in actions
            if isinstance(a, dict)
        ]
        if not action_steps:
            action_steps = ["Inspect bearing condition",
                            "Replace bearing if wear confirmed"]

        work_order = {
            # --- SAP PM Header ---
            "work_order_number":  wo_number,
            "order_type":         order_type,
            "created_at":         datetime.now(timezone.utc).isoformat(),
            "created_by":         "CARIS-AUTO",
            "plant":              "CB01",
            "planning_plant":     "CB01",

            # --- Equipment ---
            "equipment_id":       equipment,
            "equipment_name":     "Charge Gas Compressor",
            "functional_location": f"CB01-{equipment}",
            "failure_code":       failure_code,

            # --- Priority and timing ---
            "priority":           priority,
            "sap_priority_code":  sap_priority["code"],
            "priority_description": sap_priority["description"],
            "required_start":     _get_required_start(priority),
            "required_end":       _get_required_end(priority, labor_hours),

            # --- Work description ---
            "short_description":  f"{fault_type.replace('_',' ').title()} "
                                  f"fault detected on {equipment}",
            "long_description":   diagnosis,
            "work_steps":         action_steps,

            # --- Materials ---
            "parts_required":     parts,
            "estimated_labor_hrs": labor_hours,
            "crew_size":          2,

            # --- Safety ---
            "safety_notes":       safety,
            "permit_required":    "PTW",
            "loto_required":      True,
            "sop_reference":      "SOP-CGC-004",

            # --- Traceability ---
            "generated_by":       "CARIS Agent 3",
            "anomaly_score":      state.get("anomaly_score", 0),
            "confidence_pct":     state.get("confidence_pct", 0),
            "source_documents":   sources,
            "sensor_readings": {
                "vibration_rms":  state.get("vibration_rms"),
                "kurtosis":       state.get("kurtosis"),
                "crest_factor":   state.get("crest_factor"),
                "timestamp":      state.get("timestamp"),
            },
        }

        print(f"[Agent 3] Work order generated: {wo_number}")
        print(f"[Agent 3] Priority: {priority} | "
              f"Failure: {failure_code} | "
              f"Parts: {len(parts)} items")

        log_entry["work_order_number"] = wo_number
        log_entry["priority"]          = priority

        return {
            **state,
            "work_order": work_order,
            "agent_log":  state.get("agent_log", []) + [log_entry],
        }

    except Exception as e:
        print(f"[Agent 3] ERROR: {e}")
        log_entry["error"] = str(e)
        return {
            **state,
            "work_order": {"error": str(e), "priority": "P2",
                           "equipment_id": state.get("equipment_id")},
            "error":      str(e),
            "agent_log":  state.get("agent_log", []) + [log_entry],
        }


def _get_required_start(priority: str) -> str:
    """Calculate required start time based on priority."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    delta_map = {"P1": timedelta(hours=1),
                 "P2": timedelta(hours=24),
                 "P3": timedelta(hours=72)}
    start = now + delta_map.get(priority, timedelta(hours=24))
    return start.isoformat()


def _get_required_end(priority: str, labor_hours: float) -> str:
    """Calculate required end time based on start + labor."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    delta_map = {"P1": timedelta(hours=1),
                 "P2": timedelta(hours=24),
                 "P3": timedelta(hours=72)}
    end = now + delta_map.get(priority, timedelta(hours=24)) + \
          timedelta(hours=labor_hours)
    return end.isoformat()