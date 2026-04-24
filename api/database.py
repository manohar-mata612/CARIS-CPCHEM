"""
api/database.py
---------------
Smart database layer — uses Firestore on GCP, TinyDB locally.

Automatically detects environment:
  - GCP_PROJECT_ID set in env  -> uses Firestore
  - No GCP_PROJECT_ID          -> uses TinyDB (local dev)

This means the same code works locally AND in production
with zero changes. Just set the env variable.
"""

import os
import json
import threading
from datetime import datetime, timezone

GCP_PROJECT = os.getenv("GCP_PROJECT_ID", "")
USE_FIRESTORE = bool(GCP_PROJECT)

# ── TinyDB (local) ────────────────────────────────────────────
if not USE_FIRESTORE:
    from tinydb import TinyDB, Query
    DB_PATH = "api/caris_db.json"
    _lock   = threading.Lock()

    def get_db():
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        return TinyDB(DB_PATH)

    def _safe_write(table: str, data: dict) -> int:
        with _lock:
            db = get_db()
            data["saved_at"] = datetime.now(timezone.utc).isoformat()
            result = db.table(table).insert(data)
            db.close()
            return result

    def _safe_read(table: str) -> list:
        with _lock:
            db = get_db()
            result = db.table(table).all()
            db.close()
            return result

# ── Firestore (GCP production) ────────────────────────────────
else:
    from google.cloud import firestore
    _fs_client = None

    def get_firestore():
        global _fs_client
        if _fs_client is None:
            _fs_client = firestore.Client(project=GCP_PROJECT)
        return _fs_client

    def _safe_write(collection: str, data: dict) -> str:
        data["saved_at"] = datetime.now(timezone.utc).isoformat()
        db  = get_firestore()
        ref = db.collection(collection).add(data)
        return ref[1].id

    def _safe_read(collection: str) -> list:
        db   = get_firestore()
        docs = db.collection(collection).order_by(
            "saved_at",
            direction=firestore.Query.DESCENDING
        ).limit(100).stream()
        return [doc.to_dict() for doc in docs]


# ── Public API (same interface for both backends) ─────────────

def save_sensor_reading(reading: dict) -> None:
    _safe_write("sensor_readings", reading)

def save_alert(alert: dict) -> None:
    _safe_write("alerts", alert)

def save_work_order(work_order: dict) -> None:
    _safe_write("work_orders", work_order)

def save_agent_log(log_entries: list) -> None:
    for entry in log_entries:
        _safe_write("agent_logs", entry)

def get_alerts(limit: int = 20) -> list:
    data = _safe_read("alerts")
    return sorted(data, key=lambda x: x.get("saved_at", ""),
                  reverse=True)[:limit]

def get_work_orders(limit: int = 20) -> list:
    data = _safe_read("work_orders")
    return sorted(data, key=lambda x: x.get("saved_at", ""),
                  reverse=True)[:limit]

def get_agent_logs(limit: int = 50) -> list:
    data = _safe_read("agent_logs")
    return sorted(data, key=lambda x: x.get("timestamp", ""),
                  reverse=True)[:limit]

def get_sensor_readings(equipment_id: str = None,
                        limit: int = 50) -> list:
    data = _safe_read("sensor_readings")
    if equipment_id:
        data = [r for r in data
                if r.get("equipment_id") == equipment_id]
    return sorted(data, key=lambda x: x.get("saved_at", ""),
                  reverse=True)[:limit]

def acknowledge_work_order(work_order_number: str) -> bool:
    if not USE_FIRESTORE:
        from tinydb import Query as Q
        with _lock:
            db      = get_db()
            q       = Q()
            updated = db.table("work_orders").update(
                {"status": "acknowledged",
                 "acknowledged_at": datetime.now(timezone.utc).isoformat()},
                q.work_order_number == work_order_number
            )
            db.close()
            return len(updated) > 0
    else:
        db   = get_firestore()
        docs = db.collection("work_orders")\
                 .where("work_order_number", "==", work_order_number)\
                 .stream()
        updated = False
        for doc in docs:
            doc.reference.update({
                "status": "acknowledged",
                "acknowledged_at": datetime.now(timezone.utc).isoformat()
            })
            updated = True
        return updated