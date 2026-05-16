import os
import logging
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json

logger = logging.getLogger("database")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "sos_system")


def _get_client() -> MongoClient:
    """Get MongoDB client with SSL and connection pooling."""
    client = MongoClient(
        MONGODB_URI,
        retryWrites=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        socketTimeoutMS=10000,
        maxPoolSize=50,
        minPoolSize=10
    )
    return client


def _ensure_indexes():
    """Create required indexes on collections."""
    try:
        client = _get_client()
        db = client[MONGODB_DB_NAME]

        # sos_events indexes
        db.sos_events.create_index("event_uuid", unique=True)
        db.sos_events.create_index([("created_at", -1), ("acknowledged", 1), ("source_id", 1)])
        db.sos_events.create_index([("created_at", 1)], expireAfterSeconds=2592000)  # 30-day TTL

        # object_events indexes
        db.object_events.create_index([("created_at", -1)])
        db.object_events.create_index([("source_id", 1), ("created_at", -1)])

        # help_requests indexes
        db.help_requests.create_index("event_uuid")
        db.help_requests.create_index([("status", 1), ("sent_at", -1)])

        logger.info("✅ MongoDB indexes created/verified")
        client.close()
    except Exception as e:
        logger.warning(f"⚠️  Could not create indexes: {e}")


def insert_sos_event(
    event_uuid: str,
    event_type: str,
    severity: int,
    severity_name: str,
    source_id: Optional[str] = None,
    source_path: Optional[str] = None,
    location: Optional[str] = None,
    track_id: Optional[int] = None,
    image_path: Optional[str] = None,
    meta_path: Optional[str] = None,
    flags: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Insert SOS event into MongoDB. Returns event_uuid."""
    try:
        client = _get_client()
        db = client[MONGODB_DB_NAME]

        doc = {
            "event_uuid": event_uuid,
            "created_at": datetime.utcnow(),
            "event_type": event_type,
            "severity": severity,
            "severity_name": severity_name,
            "source_id": source_id,
            "source_path": source_path,
            "location": location,
            "track_id": track_id,
            "image_path": image_path,
            "meta_path": meta_path,
            "flags": flags or [],
            "extra": extra or {},
            "acknowledged": 0,
            "resolved_at": None,
            "notes": ""
        }

        result = db.sos_events.insert_one(doc)
        logger.info(f"✅ Inserted SOS event: {event_uuid}")
        client.close()
        return event_uuid

    except Exception as e:
        logger.error(f"❌ Failed to insert SOS event: {e}")
        raise


def insert_object_event(
    event_type: str,
    track_id: Optional[int] = None,
    person_track_id: Optional[int] = None,
    source_id: Optional[str] = None,
    location: Optional[str] = None,
    class_name: str = "",
    confidence: float = 0.0,
    bbox: Optional[List[float]] = None,
    image_path: Optional[str] = None,
    seconds_unattended: float = 0.0,
    meta: Optional[Dict[str, Any]] = None,
    alert_raised: bool = False,
) -> str:
    """Insert object event into MongoDB. Returns event ID."""
    try:
        client = _get_client()
        db = client[MONGODB_DB_NAME]

        doc = {
            "created_at": datetime.utcnow(),
            "event_type": event_type,
            "track_id": track_id,
            "person_track_id": person_track_id,
            "source_id": source_id,
            "location": location,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": bbox or [],
            "image_path": image_path,
            "seconds_unattended": seconds_unattended,
            "meta": meta or {},
            "alert_raised": alert_raised
        }

        result = db.object_events.insert_one(doc)
        logger.info(f"✅ Inserted object event: {str(result.inserted_id)}")
        client.close()
        return str(result.inserted_id)

    except Exception as e:
        logger.error(f"❌ Failed to insert object event: {e}")
        raise


def acknowledge_event(event_id: str, notes: str = "") -> bool:
    """Acknowledge event by UUID or ID. Returns success status."""
    try:
        client = _get_client()
        db = client[MONGODB_DB_NAME]

        # Try by event_uuid first
        result = db.sos_events.update_one(
            {"event_uuid": event_id},
            {
                "$set": {
                    "acknowledged": 1,
                    "resolved_at": datetime.utcnow(),
                    "notes": notes
                }
            }
        )

        if result.modified_count == 0:
            # Try by ObjectId
            from bson import ObjectId
            try:
                result = db.sos_events.update_one(
                    {"_id": ObjectId(event_id)},
                    {
                        "$set": {
                            "acknowledged": 1,
                            "resolved_at": datetime.utcnow(),
                            "notes": notes
                        }
                    }
                )
            except:
                pass

        success = result.modified_count > 0
        if success:
            logger.info(f"✅ Acknowledged event: {event_id}")
        client.close()
        return success

    except Exception as e:
        logger.error(f"❌ Failed to acknowledge event: {e}")
        return False


def get_event_by_uuid(event_uuid: str) -> Optional[Dict]:
    """Get event by UUID. Returns event document or None."""
    try:
        client = _get_client()
        db = client[MONGODB_DB_NAME]

        event = db.sos_events.find_one({"event_uuid": event_uuid})
        if event:
            # Convert ObjectId to string for JSON serialization
            event["_id"] = str(event["_id"])
        client.close()
        return event

    except Exception as e:
        logger.error(f"❌ Failed to get event: {e}")
        return None


def recent_events(limit: int = 50, unacked_only: bool = False, hours: int = 24) -> List[Dict]:
    """Get recent events within time window. Returns list of events."""
    try:
        client = _get_client()
        db = client[MONGODB_DB_NAME]

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        query = {"created_at": {"$gte": cutoff_time}}

        if unacked_only:
            query["acknowledged"] = 0

        events = list(
            db.sos_events.find(query)
            .sort("created_at", -1)
            .limit(limit)
        )

        # Convert ObjectIds to strings
        for event in events:
            event["_id"] = str(event["_id"])

        client.close()
        return events

    except Exception as e:
        logger.error(f"❌ Failed to get recent events: {e}")
        return []


def events_summary() -> Dict:
    """Get summary of events. Returns counts by type and status."""
    try:
        client = _get_client()
        db = client[MONGODB_DB_NAME]

        # Count by event type
        type_counts = {}
        for event_type in ["fall", "hand_sos", "fall_warning"]:
            count = db.sos_events.count_documents({"event_type": event_type})
            type_counts[event_type] = count

        # Unacknowledged count
        unacked_count = db.sos_events.count_documents({"acknowledged": 0})

        # Object events count
        object_count = db.object_events.count_documents({})

        summary = {
            "by_type": type_counts,
            "unacknowledged": unacked_count,
            "objects_unattended": object_count
        }

        client.close()
        return summary

    except Exception as e:
        logger.error(f"❌ Failed to get events summary: {e}")
        return {"by_type": {}, "unacknowledged": 0, "objects_unattended": 0}


def health_check() -> bool:
    """Check MongoDB connection. Returns connection status."""
    try:
        client = _get_client()
        client.admin.command("ping")
        client.close()
        logger.info("✅ MongoDB connection OK")
        return True
    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB health check failed: {e}")
        return False


def insert_help_request(
    event_uuid: str,
    webhook_url: str,
    status: str = "SENT",
    response_code: Optional[int] = None,
    response_time_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> bool:
    """Log help request dispatch. Returns success status."""
    try:
        client = _get_client()
        db = client[MONGODB_DB_NAME]

        doc = {
            "event_uuid": event_uuid,
            "webhook_url": webhook_url,
            "status": status,
            "sent_at": datetime.utcnow(),
            "response_code": response_code,
            "response_time_ms": response_time_ms,
            "error": error,
            "responder_id": None,
            "responder_timestamp": None
        }

        db.help_requests.insert_one(doc)
        logger.info(f"✅ Logged help request: {event_uuid} -> {status}")
        client.close()
        return True

    except Exception as e:
        logger.error(f"❌ Failed to log help request: {e}")
        return False


# Initialize indexes on module load
try:
    _ensure_indexes()
except Exception as e:
    logger.warning(f"⚠️  Index initialization deferred: {e}")
