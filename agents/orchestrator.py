"""
agents/orchestrator.py
-----------------------
Agent 4 — LangGraph Orchestrator

WHAT IT DOES:
  Wires Agents 1, 2, and 3 into a stateful directed graph.
  Controls routing: which agent runs next based on current state.
  Enforces loop guards so the system never runs forever.
  Handles P1 escalation — pauses graph for human acknowledgment.

WHY LANGGRAPH (not plain LangChain):
  LangChain chains are linear: A -> B -> C, always.
  LangGraph is a directed graph with conditional edges:
    - If no anomaly: go to END (don't waste API calls)
    - If P1 fault:   escalate to human before work order
    - If P2/P3:      go directly to work order
    - If cycle > 3:  break the loop, go to END
  This conditional routing is impossible in plain chains.

LOOP GUARD:
  Without it, the monitor agent could trigger the diagnostic
  agent which somehow re-triggers the monitor agent forever.
  cycle_count in AgentState tracks how many times we've looped.
  If cycle_count > MAX_CYCLES we force END regardless of state.

Usage:
  from agents.orchestrator import run_pipeline
  result = run_pipeline(sensor_reading)
"""

import os
import sys
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agents.state          import AgentState
from agents.monitor_agent  import monitor_agent
from agents.diagnostic_agent import diagnostic_agent
from agents.workorder_agent  import workorder_agent

MAX_CYCLES = 3  # loop guard — never run more than 3 cycles


def route_after_monitor(state: AgentState) -> str:
    """
    Conditional edge after Agent 1.

    Decision logic:
      - No anomaly detected  -> END (save API calls)
      - Anomaly + P1 critical -> escalate (human loop)
      - Anomaly + P2/P3      -> diagnose (Agent 2)
      - Too many cycles      -> END (loop guard)
    """
    if state.get("cycle_count", 0) > MAX_CYCLES:
        print(f"[Orchestrator] Loop guard triggered "
              f"(cycle={state.get('cycle_count')}). Forcing END.")
        return "end"

    if not state.get("is_anomaly", False):
        print(f"[Orchestrator] No anomaly — routing to END")
        return "end"

    severity = state.get("severity", "normal")
    if severity == "critical":
        # check confidence — low confidence P1 goes to diagnose first
        confidence = state.get("confidence_pct", 0)
        if confidence >= 70:
            print(f"[Orchestrator] HIGH CONFIDENCE P1 — escalating")
            return "escalate"
        else:
            print(f"[Orchestrator] Low confidence critical — diagnosing first")
            return "diagnose"

    print(f"[Orchestrator] Anomaly detected (severity={severity}) — diagnosing")
    return "diagnose"


def route_after_diagnosis(state: AgentState) -> str:
    """
    Conditional edge after Agent 2.

    P1 after diagnosis -> still escalate to human
    P2/P3             -> generate work order
    """
    priority = state.get("final_priority", "P2")

    if priority == "P1":
        print(f"[Orchestrator] P1 confirmed by diagnosis — escalating")
        return "escalate"

    print(f"[Orchestrator] {priority} — generating work order")
    return "work_order"


def escalate_node(state: AgentState) -> AgentState:
    """
    Human-in-the-loop escalation node for P1 alerts.

    In production: this pauses the LangGraph execution,
    sends a PagerDuty/SMS alert to the on-call engineer,
    and waits for acknowledgment before continuing.

    For CARIS demo: prints the escalation and auto-acknowledges
    after a 3-second pause to simulate the interrupt.
    """
    import time

    print(f"\n{'='*60}")
    print(f"  P1 ESCALATION — HUMAN ACKNOWLEDGMENT REQUIRED")
    print(f"  Equipment:  {state.get('equipment_id')}")
    print(f"  Fault:      {state.get('fault_type')}")
    print(f"  Kurtosis:   {state.get('kurtosis'):.1f}")
    print(f"  RMS:        {state.get('vibration_rms'):.2f} mm/s")
    print(f"  Diagnosis:  {state.get('diagnosis', 'See agent log')[:60]}")
    print(f"  Action:     IMMEDIATE CONTROLLED SHUTDOWN REQUIRED")
    print(f"{'='*60}\n")

    log_entry = {
        "agent":     "escalation_node",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action":    "p1_human_escalation",
        "equipment": state.get("equipment_id"),
        "auto_acknowledged": True,
    }

    time.sleep(2)  # simulate human response time in demo
    print(f"[Orchestrator] P1 acknowledged — proceeding to work order")

    return {
        **state,
        "human_escalated": True,
        "final_priority":  "P1",
        "agent_log": state.get("agent_log", []) + [log_entry],
    }


def build_graph():
    """
    Construct the LangGraph StateGraph.

    Graph structure:
      START
        |
      monitor_agent (Agent 1)
        |
      [route_after_monitor]
        |-- "end"      -> END
        |-- "escalate" -> escalate_node -> work_order_agent -> END
        |-- "diagnose" -> diagnostic_agent (Agent 2)
                            |
                          [route_after_diagnosis]
                            |-- "escalate" -> escalate_node -> work_order -> END
                            |-- "work_order" -> workorder_agent (Agent 3) -> END
    """
    from langgraph.graph import StateGraph, END

    graph = StateGraph(AgentState)

    # add nodes
    graph.add_node("monitor",    monitor_agent)
    graph.add_node("diagnose",   diagnostic_agent)
    graph.add_node("escalate",   escalate_node)
    graph.add_node("work_order", workorder_agent)

    # entry point
    graph.set_entry_point("monitor")

    # conditional edges after monitor
    graph.add_conditional_edges(
        "monitor",
        route_after_monitor,
        {
            "end":      END,
            "diagnose": "diagnose",
            "escalate": "escalate",
        }
    )

    # conditional edges after diagnosis
    graph.add_conditional_edges(
        "diagnose",
        route_after_diagnosis,
        {
            "work_order": "work_order",
            "escalate":   "escalate",
        }
    )

    # escalation always goes to work order
    graph.add_edge("escalate", "work_order")

    # work order is always terminal
    graph.add_edge("work_order", END)

    return graph.compile()


def run_pipeline(sensor_reading: dict) -> dict:
    """
    Run the full CARIS agent pipeline for one sensor reading.

    Parameters
    ----------
    sensor_reading : dict with sensor features
      {
        "equipment_id":  "CB-CGC-001",
        "vibration_rms": 4.8,
        "peak":          16.2,
        "kurtosis":      11.3,
        "crest_factor":  6.8,
        "std":           2.1,
        "timestamp":     "2025-01-01T10:00:00Z"
      }

    Returns
    -------
    Final AgentState dict with all agent outputs populated.
    """
    # build initial state from sensor reading
    initial_state: AgentState = {
        "equipment_id":   sensor_reading.get("equipment_id", "CB-CGC-001"),
        "vibration_rms":  float(sensor_reading.get("vibration_rms",
                                sensor_reading.get("rms", 0))),
        "peak":           float(sensor_reading.get("peak", 0)),
        "kurtosis":       float(sensor_reading.get("kurtosis", 0)),
        "crest_factor":   float(sensor_reading.get("crest_factor", 0)),
        "std":            float(sensor_reading.get("std", 0)),
        "timestamp":      sensor_reading.get("timestamp",
                          datetime.now(timezone.utc).isoformat()),
        # agent output fields — start empty
        "is_anomaly":     False,
        "anomaly_score":  0.0,
        "severity":       "normal",
        "confidence_pct": 0.0,
        "fault_type":     "unknown",
        "diagnosis":      "",
        "root_causes":    [],
        "recommended_actions": [],
        "parts_required": [],
        "estimated_labor_hours": 0.0,
        "safety_notes":   "",
        "source_documents": [],
        "work_order":     {},
        "cycle_count":    0,
        "human_escalated": False,
        "final_priority": "normal",
        "agent_log":      [],
        "error":          None,
    }

    pipeline = build_graph()

    print(f"\n{'='*60}")
    print(f"CARIS PIPELINE — {sensor_reading.get('equipment_id')}")
    print(f"  RMS={sensor_reading.get('vibration_rms', sensor_reading.get('rms',0)):.3f} | "
          f"Kurtosis={sensor_reading.get('kurtosis',0):.2f} | "
          f"Timestamp={initial_state['timestamp'][:19]}")
    print(f"{'='*60}")

    final_state = pipeline.invoke(initial_state)

    print(f"\n[Pipeline Complete]")
    print(f"  Anomaly:   {final_state.get('is_anomaly')}")
    print(f"  Priority:  {final_state.get('final_priority')}")
    print(f"  Work order: {final_state.get('work_order', {}).get('work_order_number', 'N/A')}")

    return final_state


if __name__ == "__main__":
    print("=== CARIS Agent Pipeline Test ===\n")

    # Test 1: normal reading — should end after Agent 1
    print("--- Test 1: Normal reading ---")
    normal = {
        "equipment_id":  "CB-CGC-001",
        "vibration_rms": 1.2,
        "peak":          4.1,
        "kurtosis":      3.1,
        "crest_factor":  3.4,
        "std":           0.7,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }
    result = run_pipeline(normal)
    print(f"Result: anomaly={result['is_anomaly']} | "
          f"priority={result['final_priority']}\n")

    # Test 2: fault reading — should run all 3 agents
    print("--- Test 2: Bearing fault reading ---")
    fault = {
        "equipment_id":  "CB-CGC-001",
        "vibration_rms": 4.8,
        "peak":          16.2,
        "kurtosis":      11.3,
        "crest_factor":  6.8,
        "std":           2.1,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }
    result = run_pipeline(fault)

    if result.get("work_order"):
        print(f"\nWork Order:")
        wo = result["work_order"]
        print(f"  Number:   {wo.get('work_order_number')}")
        print(f"  Priority: {wo.get('priority')}")
        print(f"  Failure:  {wo.get('failure_code')}")
        print(f"  Parts:    {wo.get('parts_required')}")
        print(f"  Labor:    {wo.get('estimated_labor_hours')} hrs")
        print(f"  Start by: {wo.get('required_start', '')[:19]}")

    print(f"\nAgent audit log ({len(result.get('agent_log', []))} entries):")
    for entry in result.get("agent_log", []):
        print(f"  {entry['agent']} | {entry['action']} | "
              f"{entry.get('timestamp', '')[:19]}")