import os
import json
import threading
from tinydb import TinyDB, Query
from datetime import datetime, timezone

DB_PATH = "api/caris_db.json"
_lock = threading.Lock()


def get_db() -> TinyDB:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return TinyDB(DB_PATH)


def _safe_write(table_name: str, data: dict) -> int:
    with _lock:
        db = get_db()
        data["saved_at"] = datetime.now(timezone.utc).isoformat()
        result = db.table(table_name).insert(data)
        db.close()
        return result


def _safe_read(table_name: str) -> list:
    with _lock:
        db = get_db()
        result = db.table(table_name).all()
        db.close()
        return result


def save_sensor_reading(reading: dict) -> int:
    return _safe_write("sensor_readings", reading)

def save_alert(alert: dict) -> int:
    return _safe_write("alerts", alert)

def save_work_order(work_order: dict) -> int:
    return _safe_write("work_orders", work_order)

def save_agent_log(log_entries: list) -> None:
    with _lock:
        db = get_db()
        db.table("agent_logs").insert_multiple(log_entries)
        db.close()

def get_alerts(limit: int = 20) -> list:
    data = _safe_read("alerts")
    return sorted(data, key=lambda x: x.get("saved_at", ""), reverse=True)[:limit]

def get_work_orders(limit: int = 20) -> list:
    data = _safe_read("work_orders")
    return sorted(data, key=lambda x: x.get("saved_at", ""), reverse=True)[:limit]

def get_agent_logs(limit: int = 50) -> list:
    data = _safe_read("agent_logs")
    return sorted(data, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]

def get_sensor_readings(equipment_id: str = None, limit: int = 50) -> list:
    data = _safe_read("sensor_readings")
    if equipment_id:
        data = [r for r in data if r.get("equipment_id") == equipment_id]
    return sorted(data, key=lambda x: x.get("saved_at", ""), reverse=True)[:limit]

def acknowledge_work_order(work_order_number: str) -> bool:
    with _lock:
        db = get_db()
        Q = Query()
        updated = db.table("work_orders").update(
            {"status": "acknowledged",
             "acknowledged_at": datetime.now(timezone.utc).isoformat()},
            Q.work_order_number == work_order_number
        )
        db.close()
        return len(updated) > 0