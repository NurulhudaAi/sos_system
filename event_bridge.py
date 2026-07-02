#!/usr/bin/env python3
"""
event_bridge.py — Bridge ระหว่าง Python Detection Engine → PRD Incident Schema

ทำหน้าที่:
  1. แปลง detection event (Python format) → PRD schema
  2. Map detection types: hand_sos→Gesture, fall→Fall, etc.
  3. Map severity: ตัวเลข → string
  4. Map location → zone_id
  5. Insert ตรงเข้า cctv_incidents (PRD format)
  6. เตรียมสำหรับ POST ไป IAM ingest API (Phase 2)
"""

import os
import uuid
import yaml
import logging
import requests
from pathlib import Path
from datetime import datetime, UTC
from typing import Optional, Dict, Any

logger = logging.getLogger("event_bridge")

ROOT = Path(__file__).resolve().parent

# ─── Config ───────────────────────────────────────────────────────────────────
EVENT_BRIDGE_MODE = os.getenv("EVENT_BRIDGE_MODE", "direct")  # "direct" | "api"
IAM_INGEST_URL    = os.getenv("IAM_INGEST_URL", "http://localhost:3000/api/v1/incidents/ingest")

# ─── Detection Type Mapping (Python → PRD) ────────────────────────────────────
DETECTION_TYPE_MAP = {
    "fall":         "Fall",
    "hand_sos":     "Gesture",
    "pose_sos":     "Gesture",
    "object_event": "ObjectMissing",
    # Future types
    "inactivity":   "Inactivity",
}

# ─── Severity Mapping (int → PRD string) ──────────────────────────────────────
SEVERITY_MAP = {
    0: "Low",       # LOG
    1: "Medium",    # MED
    2: "High",      # HIGH
    3: "High",      # CRITICAL (PRD ไม่มี Critical — map เป็น High)
}


# ─── Zone Mapping ─────────────────────────────────────────────────────────────

def _load_zone_mapping() -> Dict[str, Dict]:
    """โหลด zone mapping จาก config/zone_mapping.yaml"""
    mapping_path = ROOT / "config" / "zone_mapping.yaml"
    try:
        data = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
        zones = data.get("zones", [])
        default = data.get("default", {})
        # สร้าง lookup: location → zone config
        lookup = {}
        for z in zones:
            lookup[z["location"]] = z
        return {"zones": lookup, "default": default}
    except Exception as e:
        logger.warning(f"⚠️  Zone mapping load failed: {e} — using defaults")
        return {
            "zones": {},
            "default": {
                "zone_id_prefix": "Unknown",
                "type": "General",
                "escalation_policy": {
                    "high_timeout": 300,
                    "medium_timeout": 900,
                    "low_timeout": 3600,
                },
                "cooldown_window": 300,
                "dedup_window": 30,
            }
        }


class EventBridge:
    """
    Bridge ระหว่าง Python Detection Engine → PRD Incident Management

    Usage:
        bridge = EventBridge()
        bridge.emit_detection({
            "event_type":  "fall",
            "severity":    2,
            "source_id":   "camera_bedroom_1",
            "location":    "ห้องนอน",
            "zone_id":     "Bedroom-1",
            "track_id":    42,
            "image_path":  "/path/to/snapshot.jpg",
            "confidence":  0.95,
            "flags":       ["MEDICAL_COLLAPSE"],
            "extra":       {"fall_result": {...}},
        })
    """

    def __init__(self, db_module=None):
        self.mode = EVENT_BRIDGE_MODE
        self.iam_ingest_url = IAM_INGEST_URL
        self.zone_mapping = _load_zone_mapping()
        self._db = db_module

        logger.info(
            f"✅ EventBridge initialized | mode={self.mode} | "
            f"zones_loaded={len(self.zone_mapping['zones'])}"
        )

    @property
    def db(self):
        """Lazy-load database module to avoid circular imports"""
        if self._db is None:
            import database as db_mod
            self._db = db_mod
        return self._db

    # ─── Public API ───────────────────────────────────────────────────────

    def emit_detection(self, detection_event: Dict[str, Any]) -> Optional[str]:
        """
        รับ detection event จาก Python detector → แปลงเป็น PRD format → insert cctv_incidents

        Args:
            detection_event: dict with keys:
                - event_type (str): "fall", "hand_sos", "pose_sos", "object_event"
                - severity (int): 0-3
                - source_id (str): camera ID
                - location (str): Thai location name
                - zone_id (str, optional): PRD zone ID
                - track_id (int, optional): person tracking ID
                - image_path (str, optional): snapshot file path
                - confidence (float): 0.0-1.0
                - flags (list, optional): alert flags
                - extra (dict, optional): additional metadata

        Returns:
            event_uuid (str) or None on failure
        """
        event_uuid = str(uuid.uuid4())

        try:
            # Transform → PRD format
            prd_event = self._transform(detection_event, event_uuid)

            # Persist
            if self.mode == "direct":
                self._insert_incident(prd_event)
            elif self.mode == "api":
                self._post_to_iam_ingest(prd_event)
            else:
                logger.error(f"❌ Unknown bridge mode: {self.mode}")
                return None

            logger.info(
                f"✅ Detection emitted: {event_uuid} | "
                f"{detection_event.get('event_type')} → "
                f"{prd_event['detectionType']} | "
                f"zone={prd_event['zone']}"
            )
            return event_uuid

        except Exception as e:
            logger.error(f"❌ EventBridge.emit_detection failed: {e}")
            return None

    # ─── Transform ────────────────────────────────────────────────────────

    def _transform(self, event: Dict, event_uuid: str) -> Dict:
        """แปลง Python detection event → PRD format"""
        event_type = event.get("event_type", "fall")
        severity_int = event.get("severity", 0)
        location = event.get("location", "")

        # Map detection type
        detection_type = DETECTION_TYPE_MAP.get(event_type, "Fall")

        # Map severity
        severity_str = SEVERITY_MAP.get(severity_int, "Low")

        # Map zone
        zone_id = event.get("zone_id") or self._location_to_zone(location)

        return {
            "event_uuid": event_uuid,
            "detectionType": detection_type,
            "zone": zone_id,
            "confidence": float(event.get("confidence", 0.0)),
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "severity": severity_str,
            "metadata": {
                "cameraId": event.get("source_id"),
                "personCount": 1,
                "trackId": event.get("track_id"),
                "imagePath": event.get("image_path"),
                "snapshotUrl": event.get("snapshot_url"),
                "originalEventType": event_type,
                "originalSeverity": severity_int,
                "originalSeverityName": event.get("severity_name", ""),
                "flags": event.get("flags", []),
                "extra": event.get("extra", {}),
            }
        }

    def _location_to_zone(self, location: str) -> str:
        """แปลง location ภาษาไทย → zone_id"""
        if not location:
            default = self.zone_mapping.get("default", {})
            return f"{default.get('zone_id_prefix', 'Unknown')}-0"

        zone_config = self.zone_mapping["zones"].get(location)
        if zone_config:
            return zone_config["zone_id"]

        # Fallback
        default = self.zone_mapping.get("default", {})
        prefix = default.get("zone_id_prefix", "Unknown")
        safe_loc = location.replace(" ", "_")
        return f"{prefix}-{safe_loc}"

    # ─── Direct Insert (cctv_incidents with PRD schema) ───────────────────

    def _insert_incident(self, prd_event: Dict):
        """Insert ตรงเข้า cctv_incidents collection ด้วย PRD schema"""
        try:
            self.db.insert_incident(
                event_uuid=prd_event["event_uuid"],
                detection_type=prd_event["detectionType"],
                zone=prd_event["zone"],
                confidence=prd_event["confidence"],
                timestamp=prd_event["timestamp"],
                severity=prd_event["severity"],
                metadata=prd_event["metadata"],
            )
        except Exception as e:
            logger.error(f"❌ incident insert failed: {e}")
            raise

    # ─── API Post (Phase 2 — Future) ──────────────────────────────────────

    def _post_to_iam_ingest(self, prd_event: Dict):
        """POST ไปที่ IAM ingest API (Phase 2)"""
        try:
            payload = {
                "detectionType": prd_event["detectionType"],
                "zone": prd_event["zone"],
                "confidence": prd_event["confidence"],
                "timestamp": prd_event["timestamp"],
                "metadata": prd_event["metadata"],
            }

            resp = requests.post(
                self.iam_ingest_url,
                json=payload,
                timeout=5,
                headers={"Content-Type": "application/json"},
            )

            if resp.status_code in (200, 201, 202):
                logger.info(f"✅ IAM ingest accepted: {resp.status_code}")
            else:
                logger.warning(f"⚠️  IAM ingest returned {resp.status_code}: {resp.text[:200]}")
                raise RuntimeError(f"IAM ingest failed: {resp.status_code}")

        except requests.RequestException as e:
            logger.error(f"❌ IAM ingest API error: {e}")
            raise


# ─── Singleton ────────────────────────────────────────────────────────────────

_bridge_instance: Optional[EventBridge] = None


def get_bridge() -> EventBridge:
    """คืน singleton EventBridge instance"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = EventBridge()
    return _bridge_instance
