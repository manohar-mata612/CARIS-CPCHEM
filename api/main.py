"""
api/main.py
-----------
FastAPI backend for CARIS.

WHY FASTAPI:
  - Async by default — handles many sensor readings simultaneously
  - Auto-generates OpenAPI docs at /docs (show this in interview)
  - Type validation via Pydantic — bad sensor data rejected at API level
  - Same pattern Frame uses for all production microservices

ENDPOINTS:
  POST /api/login              -> get JWT token
  POST /api/sensor-reading     -> trigger agent pipeline
  GET  /api/alerts             -> list anomaly alerts
  GET  /api/work-orders        -> list work orders
  GET  /api/agent-log          -> audit trail
  POST /api/acknowledge/{id}   -> engineer acknowledges P1
  GET  /api/health             -> health check
  POST /graphql                -> GraphQL endpoint

Run:
  uvicorn api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
import os
import sys
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from api.auth     import verify_token, authenticate_user, create_token
from api.database import (save_sensor_reading, save_alert, save_work_order,
                           save_agent_log, get_alerts, get_work_orders,
                           get_agent_logs, get_sensor_readings,
                           acknowledge_work_order)
@asynccontextmanager
async def lifespan(app):
    from api.startup import download_from_gcs
    download_from_gcs()
    yield

app = FastAPI(
    title="CARIS API",
    lifespan=lifespan,
    description="Cedar Bayou Agentic Reliability Intelligence System — CPChem",
    version="1.0.0",
)

# allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001",
                     "https://revenue-intelligence-app.web.app",
    "https://revenue-intelligence-app.firebaseapp.com",
    "https://caris-api-430431660680.us-central1.run.app",],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# Pydantic models — input validation
# ─────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class SensorReading(BaseModel):
    """
    One sensor reading from the Cedar Bayou sensor stream.
    Pydantic validates types and rejects bad data before
    it ever reaches the agents.
    """
    equipment_id:  str
    vibration_rms: float
    peak:          float
    kurtosis:      float
    crest_factor:  float
    std:           float
    timestamp:     Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "equipment_id":  "CB-CGC-001",
                "vibration_rms": 4.8,
                "peak":          16.2,
                "kurtosis":      11.3,
                "crest_factor":  6.8,
                "std":           2.1,
            }
        }


# ─────────────────────────────────────────
# Auth endpoints
# ─────────────────────────────────────────

@app.post("/api/login", tags=["auth"])
def login(request: LoginRequest):
    """
    Authenticate and get JWT token.
    Demo credentials: engineer / cpchem2025
    """
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(user["username"], user["role"])
    return {
        "access_token": token,
        "token_type":   "bearer",
        "username":     user["username"],
        "role":         user["role"],
        "expires_in":   "8 hours",
    }


# ─────────────────────────────────────────
# Core sensor endpoint — triggers agents
# ─────────────────────────────────────────

@app.post("/api/sensor-reading-public", tags=["scheduler"])
async def ingest_sensor_reading_public(
    reading: SensorReading,
    background_tasks: BackgroundTasks,
):
    """Public endpoint for Cloud Scheduler — no JWT required."""
    reading_dict = reading.model_dump()
    reading_dict["timestamp"] = reading_dict.get("timestamp") or \
                                 datetime.now(timezone.utc).isoformat()
    reading_dict["submitted_by"] = "cloud_scheduler"
    save_sensor_reading(reading_dict)
    background_tasks.add_task(_run_agents, reading_dict)
    return {"status": "accepted", "source": "scheduler"}

@app.post("/api/sensor-reading", tags=["agents"])
async def ingest_sensor_reading(
    reading: SensorReading,
    background_tasks: BackgroundTasks,
    token: dict = Depends(verify_token),
):
    """
    Receive one sensor reading and trigger the agent pipeline.

    WHY BACKGROUND TASKS:
      The agent pipeline takes 5-15 seconds (RAG + LLM call).
      We don't want the HTTP request to hang for 15 seconds.
      BackgroundTasks runs the pipeline after the response is sent.
      The React dashboard polls /api/alerts to see the result.

    This is the endpoint the React dashboard calls when the
    engineer clicks "Simulate Sensor Reading".
    In production, the sensor stream calls this automatically.
    """
    reading_dict = reading.model_dump()
    reading_dict["timestamp"] = reading_dict.get("timestamp") or \
                                 datetime.now(timezone.utc).isoformat()
    reading_dict["submitted_by"] = token.get("sub")

    # save raw reading immediately
    save_sensor_reading(reading_dict)

    # run agent pipeline in background
    background_tasks.add_task(_run_agents, reading_dict)

    return {
        "status":       "accepted",
        "message":      "Sensor reading received. Agent pipeline running.",
        "equipment_id": reading.equipment_id,
        "timestamp":    reading_dict["timestamp"],
        "note":         "Poll /api/alerts in 5-10 seconds for results",
    }


def _run_agents(reading: dict):
    """
    Background task — runs the full agent pipeline.
    Saves results to database.
    """
    try:
        from agents.orchestrator import run_pipeline
        result = run_pipeline(reading)

        # save alert if anomaly detected
        if result.get("is_anomaly"):
            alert = {
                "equipment_id":   result.get("equipment_id"),
                "severity":       result.get("severity"),
                "fault_type":     result.get("fault_type"),
                "anomaly_score":  result.get("anomaly_score"),
                "confidence_pct": result.get("confidence_pct"),
                "kurtosis":       result.get("kurtosis"),
                "vibration_rms":  result.get("vibration_rms"),
                "diagnosis":      result.get("diagnosis", ""),
                "priority":       result.get("final_priority"),
                "timestamp":      result.get("timestamp"),
                "status":         "open",
            }
            save_alert(alert)

        # save work order if generated
        if result.get("work_order") and result["work_order"].get("work_order_number"):
            save_work_order(result["work_order"])

        # save agent log
        if result.get("agent_log"):
            save_agent_log(result["agent_log"])

    except Exception as e:
        print(f"[API] Agent pipeline error: {e}")
        save_alert({
            "equipment_id": reading.get("equipment_id"),
            "severity":     "error",
            "fault_type":   "pipeline_error",
            "diagnosis":    str(e),
            "priority":     "P2",
            "status":       "error",
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        })


# ─────────────────────────────────────────
# Read endpoints — React dashboard polls these
# ─────────────────────────────────────────

@app.get("/api/alerts", tags=["dashboard"])
def list_alerts(
    limit: int = 20,
    token: dict = Depends(verify_token)
):
    """Get latest anomaly alerts. React polls this every 5 seconds."""
    return {"alerts": get_alerts(limit), "count": limit}


@app.get("/api/work-orders", tags=["dashboard"])
def list_work_orders(
    limit: int = 20,
    token: dict = Depends(verify_token)
):
    """Get latest AI-generated work orders."""
    return {"work_orders": get_work_orders(limit), "count": limit}


@app.get("/api/agent-log", tags=["dashboard"])
def list_agent_log(
    limit: int = 50,
    token: dict = Depends(verify_token)
):
    """Get agent decision audit trail. Shows every agent action."""
    return {"logs": get_agent_logs(limit), "count": limit}


@app.get("/api/sensor-readings", tags=["dashboard"])
def list_sensor_readings(
    equipment_id: Optional[str] = None,
    limit: int = 50,
    token: dict = Depends(verify_token)
):
    """Get recent sensor readings for dashboard heatmap."""
    return {"readings": get_sensor_readings(equipment_id, limit)}


@app.post("/api/acknowledge/{work_order_number}", tags=["dashboard"])
def acknowledge(
    work_order_number: str,
    token: dict = Depends(verify_token)
):
    """
    Engineer acknowledges a work order.
    For P1 alerts this is the human-in-the-loop confirmation.
    """
    success = acknowledge_work_order(work_order_number)
    if not success:
        raise HTTPException(status_code=404, detail="Work order not found")
    return {
        "status":             "acknowledged",
        "work_order_number":  work_order_number,
        "acknowledged_by":    token.get("sub"),
        "acknowledged_at":    datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────
# GraphQL endpoint — agents use this to query plant state
# ─────────────────────────────────────────

import strawberry
from strawberry.fastapi import GraphQLRouter


@strawberry.type
class AlertType:
    equipment_id:  str
    severity:      str
    fault_type:    str
    priority:      str
    diagnosis:     str
    timestamp:     str


@strawberry.type
class WorkOrderType:
    work_order_number: str
    equipment_id:      str
    priority:          str
    failure_code:      str
    short_description: str


@strawberry.type
class Query:
    @strawberry.field
    def recent_alerts(self, limit: int = 10) -> list[AlertType]:
        """GraphQL query — agents call this to check current alert state."""
        alerts = get_alerts(limit)
        return [AlertType(
            equipment_id = a.get("equipment_id", ""),
            severity     = a.get("severity", ""),
            fault_type   = a.get("fault_type", ""),
            priority     = a.get("priority", ""),
            diagnosis    = a.get("diagnosis", ""),
            timestamp    = a.get("timestamp", ""),
        ) for a in alerts]

    @strawberry.field
    def recent_work_orders(self, limit: int = 10) -> list[WorkOrderType]:
        """GraphQL query — get latest work orders."""
        orders = get_work_orders(limit)
        return [WorkOrderType(
            work_order_number = o.get("work_order_number", ""),
            equipment_id      = o.get("equipment_id", ""),
            priority          = o.get("priority", ""),
            failure_code      = o.get("failure_code", ""),
            short_description = o.get("short_description", ""),
        ) for o in orders]


schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")


# ─────────────────────────────────────────
# Health check
# ─────────────────────────────────────────

@app.get("/api/health", tags=["system"])
def health():
    """Health check endpoint for Cloud Run and load balancer."""
    return {
        "status":    "healthy",
        "service":   "CARIS API",
        "plant":     "cedar_bayou",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/", tags=["system"])
def root():
    return {
        "service": "CARIS — Cedar Bayou Agentic Reliability Intelligence System",
        "docs":    "/docs",
        "graphql": "/graphql",
        "health":  "/api/health",
    }