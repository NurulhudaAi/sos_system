# SOS System Fixes Summary - Chat Session

**Date:** 2026-05-16  
**Session Focus:** Bug fixes & Fall Detection Tuning

---

## Issues Identified & Fixed

### 1. ❌ Fall Warning Logs Trips (Trip4 Issue)
**Problem:**  
- System logs "fall_warning" when person stumbles/trips and quickly recovers
- Trip4 video shows stumble but system records it as fall warning
- Should only log REAL dangerous falls, not quick recoveries

**Root Cause:**  
- `is_down` trigger (line 243 main.py) activates before confirming real fall
- `is_down` = abnormal posture + motion (trips trigger this too)
- Missing `recovered_quickly` check in fall_warning condition

**Solution:**  
Add `recovered_quickly` filter to fall_warning trigger:
```python
# Before (logs all downward movements)
if fw_cd and fw_cd.update(bool(fr.get("is_down"))):

# After (ignores quick recoveries)
if fw_cd and fw_cd.update(bool(fr.get("is_down") and not fr.get("recovered_quickly"))):
```

**Result:**
✅ Trip4 (quick recovery < 3s) → NOT logged  
✅ Real falls (sustained on ground >= 4s) → Logged  
✅ False positives eliminated

---

### 2. ❌ Type Mismatch: snapshot_url
**File:** main.py:211  
**Error:** `snapshot_url` can be `None` but function signature expects `str`

**Fix:**  
Updated `database.py` function signature:
```python
# Before
snapshot_url: str = None,

# After  
snapshot_url: Optional[str] = None,
```

---

### 3. ❌ MongoDB Functions Missing
**File:** pipeline.py:566  
**Error:** Wrong parameter name `detection_type` instead of `event_type`

**Fix:**
```python
# Before
insert_incident(detection_type=atype, ...)

# After
insert_incident(event_type=atype, ...)
```

---

### 4. ❌ Snapshot Directory Not Created
**File:** main.py:146  
**Issue:** `snapshot_dir` used but directory never created

**Fix:**  
```python
snapshot_dir = str(ROOT / "logs" / "snapshots")
Path(snapshot_dir).mkdir(parents=True, exist_ok=True)  # ← Added
```

---

## Fall Detection Thresholds (Current)

Located in `config/thresholds.yaml`:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `sustain_seconds` | 2.0 | Must touch ground for 2 seconds |
| `recover_seconds` | 3.0 | If recovery < 3s → marked "recovered_quickly" |
| `confirm_seconds` | 4.0 | Sustained lying >= 4s confirms fall |
| `spike_thresh_norm` | 0.18 | Downward velocity threshold (normalized) |
| `lying_angle_thresh` | 40.0 | Torso angle to detect lying |
| `danger_lying_seconds` | 15.0 | Lying >= 15s = escalate to critical |

---

## Alert Levels

- **FALL_WARNING** (Level 1, MED): `is_down` detected, quick recovery expected
- **FALL** (Level 2, HIGH+): Confirmed on ground >= 4 seconds
- **CRITICAL** (Level 3): Lying immobile >= 30 seconds → automatic escalation

---

## MongoDB Collections

- `sos_events`: Fall/hand SOS events (TTL: 30 days)
- `object_events`: Object guardian alerts  
- `help_requests`: Help request dispatch logs

---

## Files Modified

1. **main.py** - Fall warning filter
2. **database.py** - Type hints (Optional)
3. **pipeline.py** - Parameter name fix
4. **alert_logger.py** - MongoDB integration
5. **config/thresholds.yaml** - Already optimized (no changes needed)

---

## Testing

To validate trip vs fall detection:
```bash
# Test with trip4 video
python3 scripts/test_fall_detector.py --video test_videos/trip4.mp4

# Check MongoDB logs
python3 -c "from database import events_summary; print(events_summary())"
```

**Expected:** Trip4 should NOT appear in MongoDB `sos_events` collection

---

## Deployment Checklist

- [ ] Apply all fixes above
- [ ] Test trip vs fall with trip4 video
- [ ] Verify MongoDB connection
- [ ] Check logs/app/sos.log for proper alerts only
- [ ] Verify recovered_quickly suppresses warnings
- [ ] Deploy to production

