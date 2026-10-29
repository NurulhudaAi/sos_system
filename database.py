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
import uuid
from datetime import datetime, timedelta, UTC
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import base64
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:
    AESGCM = None

try:
    from dotenv import load_dotenv
    # Load .env from current directory or project root
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()
except ImportError:
    pass

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError

logger = logging.getLogger("database")

# ─── Config จาก .env ──────────────────────────────────────────────────────────
MONGODB_URI            = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME        = os.getenv("MONGODB_DB_NAME", "iam")
INCIDENTS_COLLECTION   = os.getenv("MONGODB_INCIDENTS_COLLECTION", "Incidents")
SNAPSHOT_SERVER_URL    = os.getenv("SNAPSHOT_SERVER_URL", "http://127.0.0.1:8000")
SOURCE_EVENTS_COLLECTION = os.getenv("MONGODB_SOURCE_EVENTS_COLLECTION", "source_events")

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
            logger.warning("tlsAllowInvalidCertificates=True (dev mode). "
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
        logger.info("MongoDB client created")
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
        db[SOURCE_EVENTS_COLLECTION].create_index([("created_at", -1)])
        db[SOURCE_EVENTS_COLLECTION].create_index([("source_id", 1), ("created_at", -1)])
        logger.info("MongoDB indexes verified (PRD schema)")
    except Exception as e:
        logger.warning(f"Index creation warning: {e}")


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
    บันทึก incident ใน PRD schema → incidents collection

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

    meta = dict(metadata or {})
    img_p = meta.get("image_path") or meta.get("imagePath")
    s_url = meta.get("snapshotUrl") or meta.get("snapshot_base64")
    if not s_url and img_p:
        if str(img_p).startswith("data:image"):
            s_url = str(img_p)
        else:
            s_url = path_to_snapshot_url(img_p)
    if s_url:
        meta["snapshotUrl"] = s_url

    # Harmonize metadata fields to match frontend expectation
    if "source_id" in meta and "cameraId" not in meta:
        meta["cameraId"] = meta["source_id"]

    raw_b64 = meta.get("snapshot_base64")
    if not raw_b64 and s_url and str(s_url).startswith("data:image"):
        try:
            raw_b64 = str(s_url).split(",", 1)[1]
        except Exception:
            raw_b64 = str(s_url)

    doc = {
        # ─── ID ───────────────────────────────────────
        "event_uuid":         event_uuid,

        # ─── Detection ────────────────────────────────
        "detectionType":      detection_type,
        "zone":               zone,
        "confidence":         float(confidence),
        "timestamp":          ts,

        # ─── Snapshot Image URL / Base64 ──────────────
        "snapshotUrl":        s_url,
        "snapshot_base64":    raw_b64,

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

        # ─── Timestamps & Audit ───────────────────────
        "createdAt":          now,
        "createdBy":          "system",
        "closedAt":           None,
        "archivedAt":         None,
        "created":            {"by": None, "at": ts},
        "updated":            {"by": None, "at": now},
        "createdAt_sys":      now,
        "updatedAt_sys":      now,

        # ─── Metadata ────────────────────────────────
        "metadata":           _bson_safe(meta),
    }

    try:
        db[INCIDENTS_COLLECTION].insert_one(doc)
        logger.info(
            f"Incident inserted: {event_uuid} | "
            f"{detection_type} | {zone} | {severity}"
        )
        return event_uuid
    except DuplicateKeyError:
        logger.warning(f"Duplicate event_uuid skipped: {event_uuid}")
        return event_uuid
    except Exception as e:
        logger.error(f"Failed to insert incident: {e}")
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
        logger.info(f"Object event inserted: {event_type} | {class_name}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Failed to insert object event: {e}")
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
        logger.info(f"Help request logged: {event_uuid} → {status}")
        return True
    except Exception as e:
        logger.error(f"Failed to log help request: {e}")
        return False


# ─── Write: Source Status ────────────────────────────────────────────────────

def insert_source_status(
    source_id: str,
    source_path: str,
    location: Optional[str] = None,
    zone_id: Optional[str] = None,
    status: str = "connected",
    port: Optional[int] = None,
    error: Optional[str] = None,
) -> bool:
    """บันทึกสถานะการเชื่อมต่อ source ลง collection แยก เพื่อไม่ปะปนกับ incident FRD"""
    try:
        db = _get_db()
        db[SOURCE_EVENTS_COLLECTION].insert_one({
            "created_at": _utcnow(),
            "source_id": source_id,
            "source_path": source_path,
            "location": location,
            "zone_id": zone_id,
            "status": status,
            "port": port,
            "error": error,
        })
        logger.info(f"Source status logged in {SOURCE_EVENTS_COLLECTION}: {source_id} | {status}")
        print(f"[DB] Source status inserted into {SOURCE_EVENTS_COLLECTION}: {source_id} | {status}")
        return True
    except Exception as e:
        logger.error(f"Failed to log source status: {e}")
        print(f"[DB] Source status insert failed: {source_id} | {e}")
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
        logger.error(f"get_event_by_uuid error: {e}")
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
        logger.error(f"acknowledge_event error: {e}")
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
        logger.error(f"recent_events error: {e}")
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
        logger.error(f"events_summary error: {e}")
        return {"by_type": {}, "open_incidents": 0, "objects_unattended": 0}


# ─── Camera Zones Management ──────────────────────────────────────────────────

def update_camera_zones(cam_id: str, zones: list) -> bool:
    """Save or update active zones for a camera in MongoDB cctv_cameras."""
    try:
        db = _get_db()
        col = db["cctv_cameras"]
        existing = col.find_one({
            "$or": [
                {"code": cam_id},
                {"id": cam_id},
                {"name": cam_id},
                {"position_note": cam_id}
            ]
        })
        if existing:
            res = col.update_one(
                {"_id": existing["_id"]},
                {"$set": {
                    "zones": _bson_safe(zones),
                    "updated_at": _utcnow()
                }}
            )
        else:
            res = col.update_one(
                {"code": cam_id},
                {"$set": {
                    "code": cam_id,
                    "name": cam_id,
                    "zones": _bson_safe(zones),
                    "status": "Active",
                    "created_at": _utcnow(),
                    "updated_at": _utcnow()
                }},
                upsert=True
            )
        logger.info(f"Updated zones for camera {cam_id}: {len(zones)} zones")
        return bool(res.acknowledged)
    except Exception as e:
        logger.error(f"Failed to update camera zones for {cam_id}: {e}")
        return False


def get_camera_zones(cam_id: str) -> list:
    """Retrieve saved zones for a camera from MongoDB cctv_cameras."""
    try:
        db = _get_db()
        col = db["cctv_cameras"]
        cam = col.find_one({
            "$or": [
                {"code": cam_id},
                {"id": cam_id},
                {"name": cam_id},
                {"position_note": cam_id}
            ]
        })
        if cam and "zones" in cam and isinstance(cam["zones"], list):
            return cam["zones"]
    except Exception as e:
        logger.debug(f"get_camera_zones error for {cam_id}: {e}")
    return []


def get_camera_signal_config(cam_id: str) -> Dict[str, bool]:
    """Retrieve signal_config (enabled/disabled incidents) for a camera from MongoDB cctv_cameras.

    Defaults if camera has no specific signal_config configured:
      fall: True, hand_sos: True, pose_sos: True, gesture: True, object: True, inactivity: False
    """
    default_cfg = {
        "fall": True,
        "hand_sos": True,
        "pose_sos": True,
        "gesture": True,
        "object": True,
        "inactivity": False,
    }
    if not cam_id:
        return default_cfg
    try:
        db = _get_db()
        col = db["cctv_cameras"]
        cam = col.find_one({
            "$or": [
                {"code": cam_id},
                {"id": cam_id},
                {"name": cam_id},
                {"position_note": cam_id}
            ]
        })
        if cam and "signal_config" in cam and isinstance(cam["signal_config"], dict):
            cfg = dict(cam["signal_config"])
            res = dict(default_cfg)
            for k, v in cfg.items():
                if v is not None:
                    res[k] = bool(v)
            if "hand_sos" in cfg:
                res["hand_sos"] = bool(cfg["hand_sos"])
                if "pose_sos" not in cfg:
                    res["pose_sos"] = bool(cfg["hand_sos"])
                if "gesture" not in cfg:
                    res["gesture"] = bool(cfg["hand_sos"])
            return res
    except Exception as e:
        logger.debug(f"get_camera_signal_config error for {cam_id}: {e}")
    return default_cfg


# ─── RTSP Crypto & Active Cameras Management ─────────────────────────────────

def encrypt_rtsp_url(plain_url: str, secret_hex: Optional[str] = None) -> Tuple[str, str]:
    """Encrypt a plain RTSP URL using AES-256-GCM for MongoDB.
    Returns (encrypted_b64, iv_b64).
    """
    if not plain_url:
        return "", ""
    secret = secret_hex or os.getenv("RTSP_SECRET", "")
    if not secret:
        raise ValueError("RTSP_SECRET is not configured in environment")
    key = bytes.fromhex(secret.strip())
    iv = os.urandom(12)  # Standard 96-bit nonce for AES-GCM
    aesgcm = AESGCM(key)
    ciphertext_with_tag = aesgcm.encrypt(iv, plain_url.encode("utf-8"), None)
    return base64.b64encode(ciphertext_with_tag).decode("utf-8"), base64.b64encode(iv).decode("utf-8")


def decrypt_rtsp_url(encrypted_b64: str, iv_b64: str, secret_hex: Optional[str] = None) -> str:
    """Decrypt an AES-256-GCM encrypted RTSP URL from MongoDB."""
    if not encrypted_b64 or not iv_b64:
        return ""
    try:
        secret = secret_hex or os.getenv("RTSP_SECRET", "")
        if not secret:
            logger.error("[decrypt_rtsp_url] RTSP_SECRET is not configured in environment")
            return ""
        key = bytes.fromhex(secret.strip())
        iv = base64.b64decode(iv_b64.strip())
        ciphertext_with_tag = base64.b64decode(encrypted_b64.strip())
        aesgcm = AESGCM(key)
        decrypted = aesgcm.decrypt(iv, ciphertext_with_tag, None)
        return decrypted.decode("utf-8")
    except Exception as e:
        logger.error(f"[decrypt_rtsp_url] Decryption failed: {e}")
        return ""


def get_active_cameras(cam_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve cameras configured to be active from MongoDB cctv_cameras.

    If cam_id is provided, retrieves that specific camera even if currently inactive (for testing).
    Decrypts their RTSP URL in-memory and returns a list of source dicts ready for pipeline.
    """
    cameras = []
    try:
        db = _get_db()
        col = db["cctv_cameras"]
        if cam_id:
            query = {
                "$or": [
                    {"code": cam_id},
                    {"name": cam_id},
                    {"position_note": cam_id}
                ]
            }
        else:
            # Active cameras only
            query = {"active": True}

        for doc in col.find(query):
            cam_code = doc.get("code") or str(doc.get("_id"))
            name = doc.get("name", cam_code)
            location = doc.get("position_note") or name

            rtsp_url = ""
            if doc.get("rtsp_url_encrypted") and doc.get("rtsp_iv"):
                rtsp_url = decrypt_rtsp_url(doc["rtsp_url_encrypted"], doc["rtsp_iv"])
            elif doc.get("rtsp_url"):
                rtsp_url = doc.get("rtsp_url")

            # Only include cameras with a valid stream path
            if not rtsp_url:
                logger.debug(f"[get_active_cameras] Skipped {cam_code}: no valid RTSP URL")
                continue

            zones = doc.get("zones", [])
            signal_config = doc.get("signal_config") or {}
            cameras.append({
                "id": cam_code,
                "code": cam_code,
                "name": name,
                "location": location,
                "path": rtsp_url,
                "rtsp_url": rtsp_url,
                "zones": zones,
                "signal_config": signal_config,
                "enabled": True,
            })
        logger.info(f"Loaded {len(cameras)} camera(s) from MongoDB cctv_cameras (filter={cam_id!r})")
    except Exception as e:
        logger.error(f"Failed to fetch cameras from DB: {e}")
    return cameras



def update_camera_status(cam_id: str, status: str, error_msg: Optional[str] = None) -> bool:
    """Update camera running status in MongoDB cctv_cameras (e.g. 'online', 'offline', 'error')."""
    try:
        db = _get_db()
        col = db["cctv_cameras"]
        update_data: Dict[str, Any] = {
            "status": status,
            "updated_at": _utcnow()
        }
        if status == "online":
            update_data["rtsp_verified_at"] = _utcnow()
        if error_msg:
            update_data["retry_state.closed_reason"] = error_msg
            update_data["retry_state.last_attempt_at"] = _utcnow()

        res = col.update_one(
            {
                "$or": [
                    {"code": cam_id},
                    {"id": cam_id},
                    {"name": cam_id},
                    {"position_note": cam_id}
                ]
            },
            {"$set": update_data}
        )
        return bool(res.matched_count > 0)
    except Exception as e:
        logger.error(f"Failed to update camera status for {cam_id}: {e}")
        return False



# ─── Health check ─────────────────────────────────────────────────────────────

def health_check() -> bool:
    try:
        _get_client().admin.command("ping")
        logger.info("MongoDB connection healthy")
        return True
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")
        return False


# ─── Startup ──────────────────────────────────────────────────────────────────
# สร้าง index ตอน import
try:
    _create_indexes(_get_db())
except Exception as e:
    logger.warning(f"Index initialization deferred: {e}")


def register_or_update_camera(
    code: str,
    rtsp_url: str,
    name: Optional[str] = None,
    position_note: Optional[str] = None,
    encrypt: bool = True,
    active: bool = True
) -> bool:
    """Insert or update camera in cctv_cameras collection."""
    try:
        db = _get_db()
        col = db["cctv_cameras"]
        doc: Dict[str, Any] = {
            "code": code,
            "name": name or code,
            "position_note": position_note or (name or code),
            "active": active,
            "updated_at": _utcnow()
        }
        if encrypt:
            enc, iv = encrypt_rtsp_url(rtsp_url)
            doc["rtsp_url_encrypted"] = enc
            doc["rtsp_iv"] = iv
            doc["rtsp_url"] = None
        else:
            doc["rtsp_url"] = rtsp_url
            doc["rtsp_url_encrypted"] = None
            doc["rtsp_iv"] = None

        col.update_one({"code": code}, {"$set": doc}, upsert=True)
        return True
    except Exception as e:
        logger.error(f"Failed to register camera {code}: {e}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Database & RTSP Security CLI Utilities")
    parser.add_argument("--encrypt", type=str, help="Encrypt a plain RTSP URL using RTSP_SECRET")
    parser.add_argument("--decrypt", nargs=2, metavar=("ENC_B64", "IV_B64"), help="Decrypt an RTSP URL using ENC_B64 and IV_B64")
    parser.add_argument("--list-cams", action="store_true", help="List active cameras from MongoDB")
    parser.add_argument("--add-cam", action="store_true", help="Register/Update a camera in MongoDB cctv_cameras")
    parser.add_argument("--code", type=str, help="Camera code (e.g. CAM_01)")
    parser.add_argument("--url", type=str, help="RTSP URL")
    parser.add_argument("--name", type=str, default="", help="Camera display name")
    parser.add_argument("--plain", action="store_true", help="Store RTSP URL as plaintext without AES encryption")

    args = parser.parse_args()

    if args.encrypt:
        enc, iv = encrypt_rtsp_url(args.encrypt)
        print("\n✅ [AES-256-GCM Encrypted Successfully]")
        print(f"rtsp_url_encrypted : \"{enc}\"")
        print(f"rtsp_iv            : \"{iv}\"")
        print("\nExample MongoDB update/insert snippet:")
        print(f'{{"$set": {{"rtsp_url_encrypted": "{enc}", "rtsp_iv": "{iv}"}}}}')

    elif args.decrypt:
        dec = decrypt_rtsp_url(args.decrypt[0], args.decrypt[1])
        print(f"\nDecrypted RTSP URL: {dec}")

    elif args.list_cams:
        cams = get_active_cameras()
        print(f"\nFound {len(cams)} active camera(s) in MongoDB:")
        for c in cams:
            print(f" - [{c.get('code')}] {c.get('name')} | Path: {c.get('path')} | Active: {c.get('active', True)}")

    elif args.add_cam:
        if not args.code or not args.url:
            print("❌ Error: --code and --url are required when using --add-cam")
            sys.exit(1)
        ok = register_or_update_camera(
            code=args.code,
            rtsp_url=args.url,
            name=args.name or args.code,
            encrypt=not args.plain,
            active=True
        )
        if ok:
            mode_str = "plaintext" if args.plain else "AES-256-GCM encrypted"
            print(f"✅ Camera '{args.code}' saved to MongoDB successfully ({mode_str}).")
        else:
            print(f"❌ Failed to save camera '{args.code}'. Check log for details.")
    else:
        parser.print_help()