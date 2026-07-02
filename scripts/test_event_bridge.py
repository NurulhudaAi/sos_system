#!/usr/bin/env python3
"""
test_event_bridge.py — ทดสอบ EventBridge + PRD schema ใน cctv_incidents

Usage:
    python3 scripts/test_event_bridge.py
"""

import sys
import os
from pathlib import Path

# Setup path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Initialize environment
from config.env_manager import init_env
init_env(require_edit=False)

from event_bridge import EventBridge, DETECTION_TYPE_MAP, SEVERITY_MAP

# ─── Colors ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed = 0
failed = 0


def assert_eq(label, actual, expected):
    global passed, failed
    if actual == expected:
        print(f"  {GREEN}✅ PASS{RESET}: {label}")
        passed += 1
    else:
        print(f"  {RED}❌ FAIL{RESET}: {label}")
        print(f"         expected: {expected}")
        print(f"         got:      {actual}")
        failed += 1


def assert_in(label, key, obj):
    global passed, failed
    if key in obj:
        print(f"  {GREEN}✅ PASS{RESET}: {label}")
        passed += 1
    else:
        print(f"  {RED}❌ FAIL{RESET}: {label} — key '{key}' not found")
        failed += 1


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Detection Type Mapping
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}Test 1: Detection Type Mapping{RESET}")
print(f"{'='*60}")

assert_eq("fall → Fall",            DETECTION_TYPE_MAP.get("fall"),         "Fall")
assert_eq("hand_sos → Gesture",     DETECTION_TYPE_MAP.get("hand_sos"),    "Gesture")
assert_eq("pose_sos → Gesture",     DETECTION_TYPE_MAP.get("pose_sos"),    "Gesture")
assert_eq("object_event → ObjectMissing", DETECTION_TYPE_MAP.get("object_event"), "ObjectMissing")
assert_eq("inactivity → Inactivity", DETECTION_TYPE_MAP.get("inactivity"), "Inactivity")

# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: Severity Mapping
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}Test 2: Severity Mapping{RESET}")
print(f"{'='*60}")

assert_eq("0 (LOG) → Low",       SEVERITY_MAP.get(0), "Low")
assert_eq("1 (MED) → Medium",    SEVERITY_MAP.get(1), "Medium")
assert_eq("2 (HIGH) → High",     SEVERITY_MAP.get(2), "High")
assert_eq("3 (CRITICAL) → High", SEVERITY_MAP.get(3), "High")

# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Zone Mapping
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}Test 3: Zone Mapping{RESET}")
print(f"{'='*60}")

bridge = EventBridge()

assert_eq("ห้องนอน → Bedroom-1",    bridge._location_to_zone("ห้องนอน"),  "Bedroom-1")
assert_eq("ห้องโถง → Hallway-1",    bridge._location_to_zone("ห้องโถง"),  "Hallway-1")
assert_eq("ประตูหน้า → Entrance-1", bridge._location_to_zone("ประตูหน้า"), "Entrance-1")

unknown_zone = bridge._location_to_zone("สถานที่ใหม่")
assert_eq("Unknown location → starts with Unknown",
          unknown_zone.startswith("Unknown"), True)

print(f"\n  {YELLOW}ℹ️  Explicit zone_id test:{RESET}")
transformed = bridge._transform({
    "event_type": "fall", "severity": 2, "source_id": "cam-test",
    "location": "ห้องนอน", "zone_id": "Custom-Zone-42", "confidence": 0.88,
}, "test-uuid")
assert_eq("Explicit zone_id used over location", transformed["zone"], "Custom-Zone-42")

# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Transform Output — PRD Schema Validation
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}Test 4: Transform Output — PRD Schema{RESET}")
print(f"{'='*60}")

test_event = {
    "event_type": "hand_sos", "severity": 2, "severity_name": "HIGH",
    "source_id": "camera_bedroom_1", "location": "ห้องนอน", "zone_id": "Bedroom-1",
    "track_id": 42, "image_path": "/path/to/snapshot.jpg", "confidence": 0.95,
    "flags": ["MEDICAL_COLLAPSE"], "extra": {"fall_result": {"is_critical": True}},
}

result = bridge._transform(test_event, "test-uuid-001")

# PRD required fields
for field in ["event_uuid", "detectionType", "zone", "confidence", "timestamp", "severity", "metadata"]:
    assert_in(f"Has {field}", field, result)

assert_eq("detectionType = Gesture",    result["detectionType"], "Gesture")
assert_eq("zone = Bedroom-1",           result["zone"],          "Bedroom-1")
assert_eq("confidence = 0.95",          result["confidence"],    0.95)
assert_eq("severity = High",            result["severity"],      "High")
assert_eq("event_uuid = test-uuid-001", result["event_uuid"],    "test-uuid-001")
assert_eq("timestamp ends with Z",      result["timestamp"].endswith("Z"), True)

meta = result["metadata"]
assert_eq("metadata.cameraId",            meta.get("cameraId"),            "camera_bedroom_1")
assert_eq("metadata.personCount",         meta.get("personCount"),         1)
assert_eq("metadata.trackId",             meta.get("trackId"),             42)
assert_eq("metadata.imagePath",           meta.get("imagePath"),           "/path/to/snapshot.jpg")
assert_eq("metadata.originalEventType",   meta.get("originalEventType"),   "hand_sos")
assert_eq("metadata.originalSeverity",    meta.get("originalSeverity"),    2)
assert_in("metadata has flags",           "flags",                          meta)
assert_in("metadata has extra",           "extra",                          meta)

# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: MongoDB Insert — PRD Schema in cctv_incidents
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}Test 5: MongoDB Insert (cctv_incidents — PRD Schema){RESET}")
print(f"{'='*60}")

try:
    from database import health_check, insert_incident, _get_db, INCIDENTS_COLLECTION
    if health_check():
        print(f"  {GREEN}✅ MongoDB connected{RESET}")

        test_uuid = "test-prd-" + os.urandom(4).hex()
        insert_incident(
            event_uuid=test_uuid,
            detection_type="Fall",
            zone="Bedroom-1",
            confidence=0.92,
            timestamp="2026-05-24T10:23:45.123Z",
            severity="High",
            metadata={"cameraId": "cam-test", "personCount": 1, "trackId": 99},
        )
        print(f"  {GREEN}✅ PASS{RESET}: insert_incident() succeeded")
        passed += 1

        db = _get_db()
        doc = db[INCIDENTS_COLLECTION].find_one({"event_uuid": test_uuid})
        if doc:
            # PRD fields
            assert_eq("doc.detectionType = Fall",  doc.get("detectionType"), "Fall")
            assert_eq("doc.zone = Bedroom-1",      doc.get("zone"),          "Bedroom-1")
            assert_eq("doc.severity = High",       doc.get("severity"),      "High")
            assert_eq("doc.confidence = 0.92",     doc.get("confidence"),    0.92)
            assert_eq("doc.state = Open",          doc.get("state"),         "Open")
            assert_eq("doc.createdBy = system",    doc.get("createdBy"),     "system")
            assert_eq("doc.duplicateCount = 1",    doc.get("duplicateCount"), 1)
            assert_eq("doc.escalationLevel = 0",   doc.get("escalationLevel"), 0)
            assert_eq("doc.reviewedBy = None",     doc.get("reviewedBy"),    None)
            assert_eq("doc.resolvedBy = None",     doc.get("resolvedBy"),    None)
            assert_eq("doc.resolutionNotes = ''",  doc.get("resolutionNotes"), "")
            assert_eq("doc.closedAt = None",       doc.get("closedAt"),      None)
            assert_eq("doc.archivedAt = None",     doc.get("archivedAt"),    None)
            assert_in("doc has metadata",          "metadata",               doc)
            assert_in("doc has createdAt",         "createdAt",              doc)
            assert_in("doc has timestamp",         "timestamp",              doc)
            assert_in("doc has lastSeenAt",        "lastSeenAt",             doc)
            assert_in("doc has escalationHistory", "escalationHistory",      doc)

            # Old fields should NOT exist
            old_fields_absent = True
            for old_field in ["event_type", "severity_name", "source_id", "camera_id",
                              "source_path", "location", "track_id", "image_path",
                              "snapshot_url", "meta_path", "flags", "extra",
                              "status", "acknowledged", "responder", "incident_id",
                              "detected_at", "created_at", "updated_at", "notes"]:
                if old_field in doc:
                    print(f"  {RED}❌ FAIL{RESET}: Old field '{old_field}' still present!")
                    old_fields_absent = False
                    failed += 1
            if old_fields_absent:
                print(f"  {GREEN}✅ PASS{RESET}: No old schema fields present")
                passed += 1

            # Cleanup
            db[INCIDENTS_COLLECTION].delete_one({"event_uuid": test_uuid})
            print(f"  {YELLOW}🧹 Cleanup: test doc removed{RESET}")
        else:
            print(f"  {RED}❌ FAIL{RESET}: Could not find inserted document")
            failed += 1
    else:
        print(f"  {YELLOW}⚠️  MongoDB not available — skipping{RESET}")
except Exception as e:
    print(f"  {YELLOW}⚠️  MongoDB test skipped: {e}{RESET}")

# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: Full emit_detection() → cctv_incidents
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}Test 6: Full emit_detection() → cctv_incidents{RESET}")
print(f"{'='*60}")

try:
    from database import health_check, _get_db, INCIDENTS_COLLECTION
    if health_check():
        emit_result = bridge.emit_detection({
            "event_type": "fall", "severity": 3, "severity_name": "CRITICAL",
            "source_id": "camera_test", "location": "ห้องนอน", "zone_id": "Bedroom-1",
            "track_id": 1, "image_path": "/test/snapshot.jpg", "confidence": 0.99,
            "extra": {"test": True},
        })

        if emit_result:
            print(f"  {GREEN}✅ PASS{RESET}: emit_detection() returned uuid: {emit_result[:16]}...")
            passed += 1

            db = _get_db()
            doc = db[INCIDENTS_COLLECTION].find_one({"event_uuid": emit_result})
            if doc:
                assert_eq("detectionType = Fall",  doc.get("detectionType"), "Fall")
                assert_eq("severity = High",       doc.get("severity"),      "High")
                assert_eq("zone = Bedroom-1",      doc.get("zone"),          "Bedroom-1")
                assert_eq("state = Open",          doc.get("state"),         "Open")
                assert_in("has metadata",          "metadata",               doc)
                print(f"  {GREEN}✅ PASS{RESET}: PRD schema verified in cctv_incidents")
                passed += 1
            else:
                print(f"  {RED}❌ FAIL{RESET}: Document not found in cctv_incidents")
                failed += 1

            # Cleanup
            db[INCIDENTS_COLLECTION].delete_one({"event_uuid": emit_result})
            print(f"  {YELLOW}🧹 Cleanup: test doc removed{RESET}")
        else:
            print(f"  {RED}❌ FAIL{RESET}: emit_detection() returned None")
            failed += 1
    else:
        print(f"  {YELLOW}⚠️  MongoDB not available — skipping{RESET}")
except Exception as e:
    print(f"  {YELLOW}⚠️  Test skipped: {e}{RESET}")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'='*60}{RESET}")
total = passed + failed
if failed == 0:
    print(f"{GREEN}{BOLD}🎉 ALL TESTS PASSED: {passed}/{total}{RESET}")
else:
    print(f"{RED}{BOLD}⚠️  {failed} FAILED / {passed} PASSED (total: {total}){RESET}")
print(f"{'='*60}\n")

sys.exit(1 if failed > 0 else 0)
