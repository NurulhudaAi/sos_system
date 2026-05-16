# 📖 API & Configuration Reference

## Database API

### `insert_sos_event()`
```python
insert_sos_event(
    event_uuid: str,           # Unique ID
    event_type: str,           # 'fall' | 'hand_sos' | 'fall_warning'
    severity: int,             # 0-3 (LOG, MED, HIGH, CRITICAL)
    severity_name: str,        # 'LOG', 'MED', 'HIGH', 'CRITICAL'
    source_id: Optional[str],  # Camera ID
    source_path: Optional[str],# Video file path
    location: Optional[str],   # Thai location name
    track_id: Optional[int],   # Person tracking ID
    image_path: Optional[str], # Snapshot path
    meta_path: Optional[str],  # Metadata JSON path
    flags: List[str],          # Alert flags
    extra: Dict[str, Any],     # Additional metadata
) -> str                       # Returns: event_uuid
```

### `insert_object_event()`
```python
insert_object_event(
    event_type: str,           # 'object_appeared' | 'object_left' | 'object_taken'
    track_id: Optional[int],   # Object tracking ID
    person_track_id: Optional[int], # Person who handled object
    source_id: Optional[str],  # Camera ID
    location: Optional[str],   # Location
    class_name: str,           # YOLO class (backpack, suitcase, etc)
    confidence: float,         # Detection confidence
    bbox: List[float],         # [x1, y1, x2, y2]
    image_path: Optional[str], # Snapshot
    seconds_unattended: float, # Time unattended
    meta: Dict[str, Any],      # Metadata
    alert_raised: bool = False,# Alert flag
) -> str                       # Returns: event ID
```

### `acknowledge_event()`
```python
acknowledge_event(
    event_id: int,             # Event ID or UUID
    notes: str = ""            # Acknowledgement notes
) -> None
```

### `get_event_by_uuid()`
```python
get_event_by_uuid(
    event_uuid: str            # Event UUID
) -> Optional[Dict]            # Returns: Event document or None
```

### `recent_events()`
```python
recent_events(
    limit: int = 50,           # Max results
    unacked_only: bool = False,# Only unacknowledged
    hours: int = 24            # Time window
) -> List[Dict]                # Returns: List of events
```

### `events_summary()`
```python
events_summary() -> Dict
# Returns:
# {
#     "by_type": {"fall": 5, "hand_sos": 2},
#     "unacknowledged": 3,
#     "objects_unattended": 1
# }
```

### `health_check()`
```python
health_check() -> bool         # Returns: MongoDB connection status
```

---

## Help Request API

### `HelpRequestDispatcher.should_send_help_request()`
```python
should_send_help_request(
    event_type: str,           # 'fall' | 'hand_sos' | etc
    severity: int              # 0-3
) -> bool
# Logic:
# - hand_sos: Always True (all severities)
# - fall: True if severity >= 2 (HIGH or CRITICAL)
# - others: False
```

### `HelpRequestDispatcher.dispatch_help_request()`
```python
dispatch_help_request(
    event_uuid: str,
    event_type: str,
    severity: int,
    severity_name: str,
    location: str,
    source_id: str,
    track_id: int,
    image_path: str,           # File path (NOT base64)
    frame=None,                # Deprecated (ignored)
    extra: Optional[Dict] = None
) -> bool                      # Returns: Success/Failure
```

---

## Environment Variables

### Required
```env
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
```
**Format:** MongoDB Atlas connection string

### Optional
```env
MONGODB_DB_NAME=cctv
HELP_WEBHOOK_URL=https://api.example.com/incidents
SOS_DB_PATH=logs/sos_events.db
FORCE_RECORD_ALERTS=false
```

---

## MongoDB Collections

### `sos_events`
```javascript
{
  "_id": ObjectId(),
  "event_uuid": "uuid-string",        // Unique
  "created_at": ISODate(),
  "event_type": "fall|hand_sos|...",
  "severity": 0-3,
  "severity_name": "LOG|MED|HIGH|CRITICAL",
  "source_id": "camera_1",
  "source_path": "/path/to/video",
  "location": "ห้องนอน",
  "track_id": 123,
  "image_path": "logs/snapshots/...",
  "meta_path": "logs/snapshots/...",
  "flags": ["FLAG1", "FLAG2"],
  "extra": {...},
  "acknowledged": 0|1,
  "resolved_at": ISODate()|null,
  "notes": "string"
}
```

**Indexes:**
- `event_uuid` (unique)
- `created_at` (TTL: 30 days)
- `event_type, acknowledged, source_id`

### `object_events`
```javascript
{
  "_id": ObjectId(),
  "created_at": ISODate(),
  "event_type": "object_appeared|object_left|...",
  "track_id": 123,
  "person_track_id": 456,
  "source_id": "camera_1",
  "location": "ห้องนอน",
  "class_name": "backpack",
  "confidence": 0.95,
  "bbox": [x1, y1, x2, y2],
  "image_path": "logs/snapshots/...",
  "seconds_unattended": 120.5,
  "meta": {...},
  "alert_raised": true|false
}
```

### `help_requests`
```javascript
{
  "_id": ObjectId(),
  "event_uuid": "uuid-string",
  "webhook_url": "https://...",
  "status": "SENT|FAILED|ERROR|ACKNOWLEDGED",
  "sent_at": ISODate(),
  "response_code": 200,
  "response_time_ms": 125,
  "event_type": "fall|hand_sos",
  "severity": 0-3,
  "location": "ห้องนอน",
  "responder_id": "dispatcher_123",
  "responder_timestamp": ISODate()|null,
  "error": "error message if FAILED"
}
```

**Indexes:**
- `event_uuid`
- `status, sent_at` (for polling)

---

## Configuration Files

### `thresholds.yaml`

**Fall Detection:**
```yaml
fall:
  bbox_ratio_thresh: 1.6         # Bounding box ratio
  spine_angle_thresh: 50.0       # Degrees
  warning_seconds: 2.0           # Initial warning
  confirm_seconds: 4.0           # Confirmation time
  sustain_seconds: 2.0           # Ground contact
  recover_seconds: 3.0           # Recovery time
  motion_required_for_down: true # Require motion
```

**Hand SOS:**
```yaml
hand_sos:
  min_detection_confidence: 0.75
  min_tracking_confidence: 0.50
  min_kp_conf: 0.50              # Keypoint confidence
  frame_persist_hand: 2           # Frames for persistence
```

**Danger Assessment:**
```yaml
danger:
  immobile_seconds: 5             # Immobile threshold
  record_threshold_level: 1       # MIN level to record (MED)
  environment_modifier:
    outdoor_road_work: 1          # Add level
    elderly_child: 1
```

### `sources.yaml`

```yaml
cameras:
  - id: camera_1
    path: /path/to/video.mp4
    location_th: "ห้องนอน"
    location: "Bedroom"
    port: 8081
```

### `.env`

```env
MONGODB_URI=mongodb+srv://cctv:PASSWORD@cluster0.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=cctv
HELP_WEBHOOK_URL=
```

---

## Webhook Payload Format

### Request
```bash
POST ${HELP_WEBHOOK_URL}
Content-Type: application/json
Timeout: 5 seconds
```

### Body
```json
{
  "event_uuid": "uuid-123",
  "event_type": "hand_sos",
  "severity": 1,
  "severity_name": "MED",
  "location": "ห้องนอน",
  "timestamp": "2026-05-11T14:30:52.123456Z",
  "track_id": 123,
  "source_id": "camera_1",
  "image_path": "logs/snapshots/2026-05-11/hand_sos/...",
  "status": "PENDING_RESPONSE",
  "meta": {
    "field1": "value1",
    "field2": 123
  }
}
```

**Size:** Typical 5-10 KB (image path only)

### Expected Response
```json
{
  "status": "received",
  "message": "Help request received",
  "request_id": "req-123"
}
```

**Accepted Codes:** 200, 201, 202, 204

---

## Event Type Reference

### Detection Events
| Type | Source | Severity | Webhook |
|------|--------|----------|---------|
| `hand_sos` | Hand detector | Any | ✅ YES |
| `fall` | Fall detector | 0-3 | ✅ If ≥2 |
| `fall_warning` | Fall detector | 0 | ❌ NO |

### Object Events
| Type | Description |
|------|-------------|
| `object_appeared` | New object detected |
| `object_left` | Object left behind |
| `object_taken` | Object taken |
| `owner_left` | Owner left object |

---

## Severity Levels

| Level | Name | Description | Triggers Help |
|-------|------|-------------|---------------|
| 0 | LOG | Informational event | ❌ No |
| 1 | MED | Medium severity | ❌ No (fall only) |
| 2 | HIGH | High severity | ✅ Yes (fall+SOS) |
| 3 | CRITICAL | Critical severity | ✅ Yes (fall+SOS) |

---

## Alert Flags

Common flags assigned during assessment:
- `BALANCE_STUMBLE` - Quick recovery
- `MEDICAL_COLLAPSE` - Slow fall pattern
- `IMPACT_FALL` - High velocity
- `ELDERLY_PRIORITY` - Elderly person
- `CHILD_PRIORITY` - Child person
- `ROAD_ENVIRONMENT` - Outdoor
- `WORKPLACE_HAZARD` - Work area
- `ALONE_NO_HELP` - No helper nearby

---

## Logging Formats

### JSONL (logs/events/)
```json
{"event_uuid": "...", "created_at": "...", "event_type": "...", ...}
```

### Unified Log (logs/app/sos.log)
```
2026-05-11 14:30:52 | FALL DETECTED | severity=CRITICAL | location=ห้องนอน | img=path
```

---

## Error Codes

| Code | Meaning | Resolution |
|------|---------|------------|
| 1 | MONGODB_URI not set | Add to .env |
| 2 | Connection timeout | Check IP whitelist |
| 3 | Authentication failed | Check credentials |
| 4 | Collection not found | Run setup_mongodb.py |

