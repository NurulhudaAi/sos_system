#!/usr/bin/env python3
"""
alert_logger.py
บันทึก SOS และ Object events ลง:
  1. JSONL file  (logs/alerts.jsonl)
  2. MongoDB     (iam → cctv_incidents)
"""
import json
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

LOG_DIR  = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
JSONL_PATH = LOG_DIR / "alerts.jsonl"

_lock = threading.Lock()


# ─── JSONL helper ─────────────────────────────────────────────────────────────

def _write_jsonl(record: dict):
    """เขียน 1 บรรทัดลงไฟล์ JSONL อย่าง thread-safe"""
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _lock:
            with open(JSONL_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        logger.error(f"[alert_logger] JSONL write error: {e}")


# ─── MongoDB helper ───────────────────────────────────────────────────────────

def _write_mongo(collection: str, record: dict):
    """บันทึกลง MongoDB — ถ้า DB ล่มให้ข้ามโดยไม่ crash"""
    try:
        from database import _get_db
        db = _get_db()
        db[collection].insert_one(record)
    except Exception as e:
        logger.warning(f"[alert_logger] MongoDB write error ({collection}): {e}")


# ─── Public API ───────────────────────────────────────────────────────────────

class AlertLogger:

    def log_sos_event(
        self,
        event_type:    str,
        severity:      int,
        severity_name: str,
        source_id:     Optional[str],
        source_path:   Optional[str],
        location:      Optional[str],
        track_id:      Optional[int],
        image_path:    Optional[str],
        meta_path:     Optional[str],
        flags:         List[str],
        extra:         Dict[str, Any],
    ):
        """บันทึก SOS event → JSONL + MongoDB cctv_incidents"""
        now = datetime.utcnow()

        record = {
            "log_type":     "sos_event",
            "event_type":   event_type,
            "severity":     severity,
            "severity_name":severity_name,
            "source_id":    source_id,
            "source_path":  source_path,
            "location":     location,
            "track_id":     track_id,
            "image_path":   image_path,
            "meta_path":    meta_path,
            "flags":        flags,
            "extra":        extra,
            "detected_at":  now.isoformat(),
            "created_at":   now.isoformat(),
        }

        # 1. JSONL
        _write_jsonl(record)

        # 2. MongoDB
        mongo_doc = {
            **record,
            "detected_at": now,
            "created_at":  now,
            "updated_at":  now,
            "status":      "new",
            "responder": {
                "staff_id":    None,
                "accepted_at": None,
                "resolved_at": None,
            },
        }
        _write_mongo("cctv_incidents", mongo_doc)

        logger.info(f"[alert_logger] SOS logged: {event_type} | {severity_name} @ {location}")

    def log_object_event(self, event: dict):
        """บันทึก Object Guardian event → JSONL + MongoDB cctv_incidents"""
        now = datetime.utcnow()

        record = {
            "log_type":   "object_event",
            "detected_at": now.isoformat(),
            "created_at":  now.isoformat(),
            **event,
        }

        # 1. JSONL
        _write_jsonl(record)

        # 2. MongoDB
        mongo_doc = {
            **record,
            "detected_at": now,
            "created_at":  now,
            "updated_at":  now,
            "status":      "new",
            "responder": {
                "staff_id":    None,
                "accepted_at": None,
                "resolved_at": None,
            },
        }
        _write_mongo("cctv_incidents", mongo_doc)

        logger.info(
            f"[alert_logger] Object event logged: "
            f"{event.get('event_type')} | {event.get('class_name')} @ {event.get('location')}"
        )


# ─── Singleton ────────────────────────────────────────────────────────────────

alert_logger = AlertLogger()
