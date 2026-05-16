# 📚 CCTV SOS Detection System - Implementation Summary

## Project Overview

**Objective:** MongoDB-based SOS detection system with real-time help request notifications

**Stack:**
- Python 3.14
- MongoDB Atlas (Cloud)
- FastAPI (Alert dispatching)
- MediaPipe (Hand detection)
- YOLO (Pose detection)

---

## 🏗️ Architecture

```
Detection Event (Fall / Hand SOS)
    ↓
AlertDispatcher (pipeline.py)
    ├─→ Create snapshot (JPEG + JSON)
    ├─→ Write to JSONL + unified log
    ├─→ Insert to MongoDB
    └─→ Send webhook (image_path only)
```

---

## 🎯 Main Implementations

### 1. MongoDB Integration
- **File:** `database.py`
- **Database:** `cctv` (via MONGODB_DB_NAME env)
- **Collections:**
  - `sos_events` - Fall/Hand SOS detections
  - `object_events` - Object tracking
  - `help_requests` - Webhook dispatch logs

### 2. Real-Time Help Requests
- **File:** `help_request_dispatcher.py`
- **Triggers:**
  - ALL hand SOS events
  - HIGH/CRITICAL falls only (severity ≥ 2)
- **Webhook Payload:** Image path (not base64)

### 3. Environment Management
- **File:** `config/env_manager.py`
- **Features:**
  - Auto-create .env if missing
  - Validate critical variables
  - Wait for user configuration
  - Secure masking for sensitive values

### 4. Configuration
- **File:** `config/thresholds.yaml`
- Sections:
  - `fall` - Fall detection parameters
  - `hand_sos` - Hand gesture recognition
  - `danger` - Alert level assessment
  - `mongodb` - Database config
  - `help_requests` - Webhook settings

---

## 📋 File Structure

```
project/
├── .env                          ← MongoDB connection (user-created)
├── .env.example                  ← Template
├── main.py                       ← Entry point
├── database.py                   ← MongoDB client
├── config/
│   ├── env_manager.py           ← .env management
│   ├── thresholds.yaml          ← Detection parameters
│   ├── sources.yaml             ← Camera locations
│   └── zones.yaml               ← Detection zones
├── detectors/
│   ├── fall_detector.py         ← Fall detection
│   ├── hand_sos_detector.py     ← Hand SOS detection
│   ├── object_guardian.py       ← Object tracking
│   └── pose_sos_detector.py     ← Pose detection
├── pipeline.py                   ← Alert dispatcher
├── help_request_dispatcher.py   ← Webhook notifications
├── alert_logger.py              ← Event logging
├── scripts/
│   ├── setup_mongodb.py         ← Setup helper
│   ├── test_mongodb_webhook.py  ← Test script
│   └── test_mongo_quick.py      ← Quick connection test
└── logs/
    ├── app/                     ← Unified logs
    ├── events/                  ← NDJSON daily events
    ├── snapshots/               ← Alert images (dated)
    └── tuning/                  ← Parameter versions
```

---

## 🚀 Setup Process

### Step 1: Get MongoDB Connection
From MongoDB Atlas:
```
mongodb+srv://cctv:PASSWORD@cluster0.9ujh3gg.mongodb.net/?retryWrites=true&w=majority
```

### Step 2: Create .env
```env
MONGODB_URI=mongodb+srv://cctv:PASSWORD@cluster0.9ujh3gg.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=cctv
HELP_WEBHOOK_URL=
```

### Step 3: Test Connection
```bash
python3 test_mongo_quick.py
```

### Step 4: Run System
```bash
python3 main.py
```

---

## 📊 Event Severity Levels

| Level | Name | Description | Webhook |
|-------|------|-------------|---------|
| 0 | LOG | Informational | ❌ No |
| 1 | MED | Medium severity | ❌ No |
| 2 | HIGH | High severity | ✅ Yes |
| 3 | CRITICAL | Critical severity | ✅ Yes |

**Hand SOS:** Always triggers webhook (all levels)

---

## 🔧 Key Features

### MongoDB Features
- ✅ Thread-safe connection pooling
- ✅ Automatic index creation
- ✅ TTL index (auto-delete 30 days)
- ✅ SSL certificate handling (tlsAllowInvalidCertificates)
- ✅ Extended timeouts (10s)

### Help Request Features
- ✅ Configurable webhook URL
- ✅ Retry logic (3 attempts)
- ✅ Payload size optimization (<10KB)
- ✅ Dispatch status tracking
- ✅ Base64 encoding removed

### Alert System
- ✅ Unified logging (JSONL + text)
- ✅ Daily snapshot organization
- ✅ Event metadata tracking
- ✅ Danger level assessment

---

## 📝 Configuration Example

### thresholds.yaml Sections

**Fall Detection:**
```yaml
fall:
  bbox_ratio_thresh: 1.6
  spine_angle_thresh: 50.0
  sustain_seconds: 2.0
  motion_required_for_down: true
```

**Hand SOS:**
```yaml
hand_sos:
  min_detection_confidence: 0.75
  min_tracking_confidence: 0.50
  min_kp_conf: 0.50
  frame_persist_hand: 2
```

**MongoDB:**
```yaml
mongodb:
  enabled: true
  uri: "${MONGODB_URI}"
  db_name: "cctv"
  ttl_seconds: 2592000
```

**Help Requests:**
```yaml
help_requests:
  enabled: true
  webhook_url: "${HELP_WEBHOOK_URL}"
  triggers:
    - event_type: "hand_sos"
      severity: null
    - event_type: "fall"
      min_severity: 2
```

---

## 🔐 Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `MONGODB_URI` | MongoDB connection | `mongodb+srv://...` |
| `MONGODB_DB_NAME` | Database name | `cctv` |
| `HELP_WEBHOOK_URL` | Webhook endpoint | `https://api.example.com/incidents` |
| `SOS_DB_PATH` | SQLite fallback path | `logs/sos_events.db` |
| `FORCE_RECORD_ALERTS` | Force all recording | `false` |

---

## 📤 Webhook Payload Example

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
    "time_lying": 0,
    "velocity_norm": 0.25
  }
}
```

**Key:** Image sent as **path only** (not base64)

---

## 🐛 Troubleshooting

### SSL Certificate Error
**Solution:** Increase timeouts in `database.py`
```python
serverSelectionTimeoutMS=10000
connectTimeoutMS=10000
socketTimeoutMS=10000
tlsAllowInvalidCertificates=True
```

### .env Not Found
**Solution:** `env_manager.py` auto-creates with defaults, then waits for user edit

### MongoDB Not Connecting
**Checklist:**
- [ ] MONGODB_URI correct
- [ ] IP whitelist in MongoDB Atlas
- [ ] Username/password valid
- [ ] Network connectivity

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `QUICK_START_MONGODB.md` | 5-minute setup |
| `SETUP_GUIDE.md` | Detailed guide |
| `MONGODB_SETUP_CCTV.md` | MongoDB specific |
| `MONGODB_SETUP.md` | Full reference |

---

## 🎯 Next Steps

1. ✅ Run `python3 main.py`
2. ✅ Monitor MongoDB collections
3. ✅ Configure webhook URL
4. ✅ Test with video sources
5. ✅ Fine-tune detection thresholds

---

## 📞 Support

For issues, check:
1. `.env` file exists with MONGODB_URI
2. MongoDB Atlas connection string valid
3. IP whitelist in Atlas
4. Network connectivity to cluster
5. Python version 3.10+

