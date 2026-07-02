#!/usr/bin/env python3
"""
database.py — MongoDB backend สำหรับ SOS System (PRD Schema)
เชื่อมกับ Atlas cluster: iam database → cctv_incidents collection

Schema ตรงกับ PRD Section 10.1 — Incident Collection:
  detectionType, zone, state, severity, confidence,
  duplicateCount, escalationLevel, escalationHistory,
  reviewedBy, resolvedBy, metadata, etc.
"""
import os
import logging
import threading
from datetime import datetime, timedelta, UTC
from typing import Optional, Dict, Any, List

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError

logger = logging.getLogger("database")

# ─── Config จาก .env ──────────────────────────────────────────────────────────
MONGODB_URI            = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME        = os.getenv("MONGODB_DB_NAME", "iam")
INCIDENTS_COLLECTION   = os.getenv("MONGODB_INCIDENTS_COLLECTION", "cctv_incidents")
SNAPSHOT_SERVER_URL    = os.getenv("SNAPSHOT_SERVER_URL", "http://127.0.0.1:8000")

# ─── Thread-local connection pool (1 client ต่อ thread) ──────────────────────
_local = threading.local()


def path_to_snapshot_url(file_path: Optional[str]) -> Optional[str]:
    """แปลง local file path เป็น HTTP URL สำหรับ snapshot serve"""
    if not file_path:
        return None
    from pathlib import Path
    p = Path(file_path)
    filename = p.name
    return f"{SNAPSHOT_SERVER_URL}/snapshot/{filename}"


def _get_client() -> MongoClient:
    """คืน MongoClient แบบ thread-local — สร้างครั้งเดียวต่อ thread"""
    if not hasattr(_local, "client") or _local.client is None:
        # [W4] tlsAllowInvalidCertificates ปิดใน production, เปิดเฉพาะ dev
        APP_ENV = os.getenv("APP_ENV", "development")
        _tls_allow_invalid = (APP_ENV != "production")
        if _tls_allow_invalid:
            logger.warning("⚠️  tlsAllowInvalidCertificates=True (dev mode). "
                           "Set APP_ENV=production to disable.")
        _local.client = MongoClient(
            MONGODB_URI,
            tls=True,
            tlsAllowInvalidCertificates=_tls_allow_invalid,
            retryWrites=True,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=10000,
            maxPoolSize=50,
            minPoolSize=0,
        )
        logger.info("✅ MongoDB client created")
    return _local.client


def _get_db():
    """คืน database instance"""
    return _get_client()[MONGODB_DB_NAME]


def _utcnow():
    return datetime.now(UTC)


def _bson_safe(value):
    """แปลง numpy / dict / list ให้ MongoDB รับได้"""
    try:
        import numpy as np
        if isinstance(value, np.integer): return int(value)
        if isinstance(value, np.floating): return float(value)
        if isinstance(value, np.ndarray): return [_bson_safe(i) for i in value.tolist()]
    except Exception:
        pass
    if isinstance(value, dict):  return {str(k): _bson_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)): return [_bson_safe(i) for i in value]
    return value


# ─── Index setup ──────────────────────────────────────────────────────────────

def _create_indexes(db):
    """สร้าง index ที่จำเป็น — เรียกครั้งเดียวตอน startup"""
    try:
        col = db[INCIDENTS_COLLECTION]

        # PRD indexes
        col.create_index("event_uuid", unique=True, sparse=True)
        col.create_index([("zone", 1), ("detectionType", 1), ("timestamp", 1)])
        col.create_index([("state", 1), ("createdAt", -1)])
        col.create_index([("severity", 1), ("state", 1)])
        col.create_index([("createdAt", -1)])
        col.create_index([("escalationLevel", 1)])
        try:
            col.create_index([("createdAt", 1)], expireAfterSeconds=2592000)  # TTL 30 วัน
        except Exception:
            pass  # index อาจมีอยู่แล้ว

        # Other collections
        db.object_events.create_index([("created_at", -1)])
        db.object_events.create_index([("source_id", 1), ("created_at", -1)])
        db.help_requests.create_index("event_uuid")
        db.help_requests.create_index([("status", 1), ("sent_at", -1)])

        logger.info("✅ MongoDB indexes verified (PRD schema)")
    except Exception as e:
        logger.warning(f"⚠️  Index creation warning: {e}")


# ─── Write: Incident (PRD Schema) ────────────────────────────────────────────

def insert_incident(
    event_uuid:     str,
    detection_type: str,
    zone:           str,
    confidence:     float,
    timestamp:      str,
    severity:       str,
    metadata:       Optional[Dict] = None,
) -> str:
    """
    บันทึก incident ใน PRD schema → cctv_incidents collection

    Document structure ตรงกับ PRD Section 10.1:
    {
        "event_uuid":         "uuid-v4",
        "detectionType":      "Fall | Inactivity | Gesture | ObjectMissing",
        "zone":               "Hallway-B2",
        "state":              "Open | InReview | Escalated | Resolved | Closed | Archived",
        "severity":           "High | Medium | Low",
        "confidence":         0.95,
        "duplicateCount":     1,
        "lastSeenAt":         datetime,
        "escalationLevel":    0,
        "escalationHistory":  [],
        "reviewedBy":         null,
        "reviewedAt":         null,
        "resolvedBy":         null,
        "resolvedAt":         null,
        "resolutionNotes":    "",
        "timestamp":          datetime,
        "createdAt":          datetime,
        "createdBy":          "system",
        "closedAt":           null,
        "archivedAt":         null,
        "metadata":           { "cameraId": "cam-001", ... }
    }
    """
    db  = _get_db()
    now = _utcnow()

    # Parse timestamp string → datetime
    try:
        ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        ts = now

    doc = {
        # ─── ID ───────────────────────────────────────
        "event_uuid":         event_uuid,

        # ─── Detection ────────────────────────────────
        "detectionType":      detection_type,
        "zone":               zone,
        "confidence":         float(confidence),
        "timestamp":          ts,

        # ─── State Machine (PRD Section 6) ────────────
        "state":              "Open",
        "severity":           severity,

        # ─── Deduplication (PRD FR-NEW-004) ───────────
        "duplicateCount":     1,
        "lastSeenAt":         ts,

        # ─── Escalation (PRD FR-NEW-003) ──────────────
        "escalationLevel":    0,
        "escalationHistory":  [],

        # ─── Operator Workflow ────────────────────────
        "reviewedBy":         None,
        "reviewedAt":         None,
        "resolvedBy":         None,
        "resolvedAt":         None,
        "resolutionNotes":    "",

        # ─── Timestamps ──────────────────────────────
        "createdAt":          now,
        "createdBy":          "system",
        "closedAt":           None,
        "archivedAt":         None,

        # ─── Metadata ────────────────────────────────
        "metadata":           _bson_safe(metadata or {}),
    }

    try:
        db[INCIDENTS_COLLECTION].insert_one(doc)
        logger.info(
            f"✅ Incident inserted: {event_uuid} | "
            f"{detection_type} | {zone} | {severity}"
        )
        return event_uuid
    except DuplicateKeyError:
        logger.warning(f"⚠️  Duplicate event_uuid skipped: {event_uuid}")
        return event_uuid
    except Exception as e:
        logger.error(f"❌ Failed to insert incident: {e}")
        raise


# ─── Write: Object Event ──────────────────────────────────────────────────────

def insert_object_event(
    event_type:          str,
    track_id:            Optional[int]   = None,
    person_track_id:     Optional[int]   = None,
    source_id:           Optional[str]   = None,
    location:            Optional[str]   = None,
    class_name:          str             = "",
    confidence:          float           = 0.0,
    bbox:                Optional[list]  = None,
    image_path:          Optional[str]   = None,
    seconds_unattended:  float           = 0.0,
    meta:                Optional[Dict]  = None,
    alert_raised:        bool            = False,
) -> str:
    """บันทึก Object Guardian event → object_events. คืนค่า inserted_id"""
    db  = _get_db()
    now = _utcnow()

    doc = {
        "created_at":         now,
        "event_type":         event_type,
        "track_id":           track_id,
        "person_track_id":    person_track_id,
        "source_id":          source_id,
        "location":           location,
        "class_name":         class_name,
        "confidence":         confidence,
        "bbox":               bbox or [],
        "image_path":         image_path,
        "seconds_unattended": seconds_unattended,
        "meta":               _bson_safe(meta or {}),
        "alert_raised":       alert_raised,
    }

    try:
        result = db.object_events.insert_one(doc)
        logger.info(f"✅ Object event inserted: {event_type} | {class_name}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"❌ Failed to insert object event: {e}")
        raise


# ─── Write: Help Request ──────────────────────────────────────────────────────

def insert_help_request(
    event_uuid:       str,
    webhook_url:      str,
    status:           str            = "SENT",
    response_code:    Optional[int]  = None,
    response_time_ms: Optional[float]= None,
    error:            Optional[str]  = None,
) -> bool:
    """บันทึกสถานะการส่ง webhook → help_requests"""
    db = _get_db()
    try:
        db.help_requests.insert_one({
            "event_uuid":       event_uuid,
            "webhook_url":      webhook_url,
            "status":           status,
            "sent_at":          _utcnow(),
            "response_code":    response_code,
            "response_time_ms": response_time_ms,
            "error":            error,
        })
        logger.info(f"✅ Help request logged: {event_uuid} → {status}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to log help request: {e}")
        return False


# ─── Read helpers ─────────────────────────────────────────────────────────────

def get_event_by_uuid(event_uuid: str) -> Optional[Dict]:
    """ดึง incident จาก UUID"""
    db = _get_db()
    try:
        doc = db[INCIDENTS_COLLECTION].find_one({"event_uuid": event_uuid})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        logger.error(f"❌ get_event_by_uuid error: {e}")
        return None


def acknowledge_event(event_id: str, notes: str = "") -> bool:
    """Transition: state → InReview (PRD FR-NEW-002)"""
    db = _get_db()
    try:
        result = db[INCIDENTS_COLLECTION].update_one(
            {"event_uuid": event_id},
            {"$set": {
                "state":        "InReview",
                "reviewedAt":   _utcnow(),
                "resolutionNotes": notes,
            }}
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"❌ acknowledge_event error: {e}")
        return False


def recent_events(limit: int = 50, unacked_only: bool = False, hours: int = 24) -> List[Dict]:
    """ดึง incidents ล่าสุด — สำหรับ Dashboard"""
    db = _get_db()
    try:
        query: Dict[str, Any] = {"createdAt": {"$gte": _utcnow() - timedelta(hours=hours)}}
        if unacked_only:
            query["state"] = "Open"
        docs = list(
            db[INCIDENTS_COLLECTION]
            .find(query)
            .sort("createdAt", -1)
            .limit(limit)
        )
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs
    except Exception as e:
        logger.error(f"❌ recent_events error: {e}")
        return []


def events_summary() -> Dict:
    """สรุปจำนวน incidents แต่ละประเภท — สำหรับ Dashboard KPI"""
    db = _get_db()
    try:
        pipeline = [
            {"$group": {"_id": "$detectionType", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        type_counts = {r["_id"]: r["count"] for r in db[INCIDENTS_COLLECTION].aggregate(pipeline)}
        open_count  = db[INCIDENTS_COLLECTION].count_documents({"state": "Open"})
        obj_count   = db.object_events.count_documents({})
        return {
            "by_type":            type_counts,
            "open_incidents":     open_count,
            "objects_unattended": obj_count,
        }
    except Exception as e:
        logger.error(f"❌ events_summary error: {e}")
        return {"by_type": {}, "open_incidents": 0, "objects_unattended": 0}


# ─── Health check ─────────────────────────────────────────────────────────────

def health_check() -> bool:
    """เช็คว่าเชื่อม MongoDB ได้ไหม"""
    try:
        _get_client().admin.command("ping")
        logger.info("✅ MongoDB connection healthy")
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB health check failed: {e}")
        return False


# ─── Startup ──────────────────────────────────────────────────────────────────
# สร้าง index ตอน import
try:
    _create_indexes(_get_db())
except Exception as e:
    logger.warning(f"⚠️  Index initialization deferred: {e}")