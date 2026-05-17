#!/usr/bin/env python3
"""
database.py — MongoDB backend สำหรับ SOS System
เชื่อมกับ Atlas cluster: iam database → cctv_incidents collection
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
MONGODB_URI          = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME      = os.getenv("MONGODB_DB_NAME", "iam")
INCIDENTS_COLLECTION = os.getenv("MONGODB_INCIDENTS_COLLECTION", "cctv_incidents")

# ─── Thread-local connection pool (1 client ต่อ thread) ──────────────────────
_local = threading.local()


def _get_client() -> MongoClient:
    """คืน MongoClient แบบ thread-local — สร้างครั้งเดียวต่อ thread"""
    if not hasattr(_local, "client") or _local.client is None:
        _local.client = MongoClient(
            MONGODB_URI,
            tls=True,
            tlsAllowInvalidCertificates=True,   # macOS dev
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
        col.create_index("event_uuid", unique=True, sparse=True)
        col.create_index([("created_at", -1), ("acknowledged", 1), ("source_id", 1)])
        try:
            col.create_index([("created_at", 1)], expireAfterSeconds=2592000)  # TTL 30 วัน
        except Exception:
            pass  # index อาจมีอยู่แล้ว

        db.object_events.create_index([("created_at", -1)])
        db.object_events.create_index([("source_id", 1), ("created_at", -1)])
        db.help_requests.create_index("event_uuid")
        db.help_requests.create_index([("status", 1), ("sent_at", -1)])

        logger.info("✅ MongoDB indexes verified")
    except Exception as e:
        logger.warning(f"⚠️  Index creation warning: {e}")


# ─── Write: SOS Event ─────────────────────────────────────────────────────────

def insert_incident(
    event_uuid:    str,
    event_type:    str,
    severity:      int,
    severity_name: str,
    source_id:     Optional[str]       = None,
    source_path:   Optional[str]       = None,
    location:      Optional[str]       = None,
    track_id:      Optional[int]       = None,
    image_path:    Optional[str]       = None,
    meta_path:     Optional[str]       = None,
    flags:         Optional[List[str]] = None,
    extra:         Optional[Dict]      = None,
) -> str:
    """บันทึก SOS event → cctv_incidents. คืนค่า event_uuid"""
    db  = _get_db()
    now = _utcnow()

    doc = {
        # ─── ID ───────────────────────────────────────
        "event_uuid":   event_uuid,
        "incident_id":  event_uuid,

        # ─── เวลา ─────────────────────────────────────
        "detected_at":  now,
        "created_at":   now,
        "updated_at":   now,

        # ─── ประเภทและความรุนแรง ──────────────────────
        "event_type":    event_type,
        "severity":      severity,
        "severity_name": severity_name,

        # ─── กล้องและสถานที่ ──────────────────────────
        "source_id":    source_id,
        "camera_id":    source_id,
        "source_path":  source_path,
        "location":     location,
        "track_id":     track_id,

        # ─── ไฟล์ snapshot ────────────────────────────
        "image_path":   image_path,
        "snapshot_url": image_path,
        "meta_path":    meta_path,

        # ─── ข้อมูลเพิ่มเติม ─────────────────────────
        "flags":        _bson_safe(flags or []),
        "extra":        _bson_safe(extra or {}),

        # ─── สถานะ ────────────────────────────────────
        "status":       "new",
        "acknowledged": 0,
        "resolved_at":  None,
        "notes":        "",

        # ─── ผู้รับผิดชอบ ─────────────────────────────
        "responder": {
            "staff_id":    None,
            "accepted_at": None,
            "resolved_at": None,
        },
    }

    try:
        db[INCIDENTS_COLLECTION].insert_one(doc)
        logger.info(f"✅ SOS event inserted: {event_uuid} | {event_type} | {severity_name}")
        return event_uuid
    except DuplicateKeyError:
        logger.warning(f"⚠️  Duplicate event_uuid skipped: {event_uuid}")
        return event_uuid
    except Exception as e:
        logger.error(f"❌ Failed to insert SOS event: {e}")
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
    """ดึง event จาก UUID"""
    db = _get_db()
    try:
        doc = db[INCIDENTS_COLLECTION].find_one({
            "$or": [{"event_uuid": event_uuid}, {"incident_id": event_uuid}]
        })
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        logger.error(f"❌ get_event_by_uuid error: {e}")
        return None


def acknowledge_event(event_id: str, notes: str = "") -> bool:
    """Mark event เป็น acknowledged"""
    db = _get_db()
    try:
        result = db[INCIDENTS_COLLECTION].update_one(
            {"$or": [{"event_uuid": event_id}, {"incident_id": event_id}]},
            {"$set": {
                "acknowledged": 1,
                "status":       "resolved",
                "resolved_at":  _utcnow(),
                "updated_at":   _utcnow(),
                "notes":        notes,
            }}
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"❌ acknowledge_event error: {e}")
        return False


def recent_events(limit: int = 50, unacked_only: bool = False, hours: int = 24) -> List[Dict]:
    """ดึง event ล่าสุด — สำหรับ Dashboard"""
    db = _get_db()
    try:
        query: Dict[str, Any] = {"created_at": {"$gte": _utcnow() - timedelta(hours=hours)}}
        if unacked_only:
            query["acknowledged"] = 0
        docs = list(
            db[INCIDENTS_COLLECTION]
            .find(query)
            .sort("created_at", -1)
            .limit(limit)
        )
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs
    except Exception as e:
        logger.error(f"❌ recent_events error: {e}")
        return []


def events_summary() -> Dict:
    """สรุปจำนวน event แต่ละประเภท — สำหรับ Dashboard KPI"""
    db = _get_db()
    try:
        pipeline = [
            {"$group": {"_id": "$event_type", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        type_counts = {r["_id"]: r["count"] for r in db[INCIDENTS_COLLECTION].aggregate(pipeline)}
        unacked     = db[INCIDENTS_COLLECTION].count_documents({"acknowledged": 0})
        obj_count   = db.object_events.count_documents({})
        return {
            "by_type":           type_counts,
            "unacknowledged":    unacked,
            "objects_unattended": obj_count,
        }
    except Exception as e:
        logger.error(f"❌ events_summary error: {e}")
        return {"by_type": {}, "unacknowledged": 0, "objects_unattended": 0}


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