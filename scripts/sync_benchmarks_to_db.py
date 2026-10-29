#!/usr/bin/env python3
"""
scripts/sync_benchmarks_to_db.py — Sync detected benchmark events from all source videos into MongoDB Atlas.
Reads scan_summary.json and the saved high-res annotated snapshot images from disk,
encodes snapshots to base64, and calls database.insert_incident() for every event.
"""
import sys
import json
import uuid
import base64
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
from database import insert_incident, _get_db, INCIDENTS_COLLECTION


def sync_events_to_db(summary_json_path: Path):
    summary_json_path = Path(summary_json_path)
    if not summary_json_path.exists():
        print(f"❌ File not found: {summary_json_path}")
        return 0

    base_dir = summary_json_path.parent
    data = json.loads(summary_json_path.read_text(encoding="utf-8"))

    det_map = {
        "fall": "Fall",
        "hand_sos": "Gesture",
        "pose_sos": "Gesture",
        "critical_sos": "Gesture",
        "object_event": "ObjectMissing",
    }
    sev_map = {
        "critical_sos": "Critical",
        "fall": "High",
        "hand_sos": "High",
        "pose_sos": "High",
    }

    db = _get_db()
    col = db[INCIDENTS_COLLECTION]
    existing_count = col.count_documents({})
    print(f"📊 Current incidents in MongoDB ({INCIDENTS_COLLECTION}): {existing_count}")
    print(f"🚀 Processing events from {summary_json_path}...\n")

    inserted_count = 0
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    for video_name, events in data.items():
        cam_code = Path(video_name).stem
        print(f"📹 Source Video: {video_name} ({len(events)} events)")
        if not events:
            print("   (clean pass - 0 events)")
            continue

        for idx, ev in enumerate(events, 1):
            ev_type = ev.get("event_type", "fall")
            lbl = ev.get("label", ev_type.upper())
            conf = float(ev.get("confidence", 0.90))
            tid = ev.get("track_id", 0)
            ts_sec = float(ev.get("timestamp_sec", 0.0))
            bbox = ev.get("bbox", [])
            details = ev.get("details", "")

            # Load snapshot image and convert to Base64
            snap_rel = ev.get("snapshot", "")
            snap_path = base_dir / snap_rel if snap_rel else None

            b64_url = ""
            raw_b64 = ""
            if snap_path and snap_path.exists():
                img = cv2.imread(str(snap_path))
                if img is not None:
                    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    raw_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                    b64_url = f"data:image/jpeg;base64,{raw_b64}"

            ev_uuid = str(uuid.uuid4())
            detection_type = det_map.get(ev_type, "Fall")
            severity = sev_map.get(ev_type, "High")

            metadata = {
                "cameraId":          cam_code,
                "source_id":         cam_code,
                "video_file":        video_name,
                "track_id":          tid,
                "trackId":           tid,
                "snapshotUrl":       b64_url,
                "snapshot_base64":   raw_b64,
                "imagePath":         str(snap_path) if snap_path else None,
                "event_type":        ev_type,
                "label":             lbl,
                "video_timestamp_s": ts_sec,
                "bbox":              bbox,
                "details":           details,
            }

            try:
                insert_incident(
                    event_uuid     = ev_uuid,
                    detection_type = detection_type,
                    zone           = cam_code,
                    confidence     = conf,
                    timestamp      = now_iso,
                    severity       = severity,
                    metadata       = metadata,
                )
                inserted_count += 1
                print(f"   [{idx}/{len(events)}] 💾 Inserted: {detection_type} ({severity}) - {lbl} @ {ts_sec}s -> uuid={ev_uuid[:8]}...")
            except Exception as e:
                print(f"   [{idx}/{len(events)}] ❌ Failed to insert: {e}")

        print()

    new_count = col.count_documents({})
    print(f"{'='*60}")
    print(f"🎉 Successfully inserted {inserted_count} incident(s) into MongoDB Atlas!")
    print(f"📊 Total incidents in collection '{INCIDENTS_COLLECTION}': {new_count}")
    print(f"{'='*60}")
    return inserted_count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sync benchmarked events to MongoDB Atlas")
    parser.add_argument("--summary", type=str, default="eval/scan_benchmark_dual_approach/scan_summary.json",
                        help="Path to scan_summary.json")
    args = parser.parse_args()

    sync_events_to_db(Path(args.summary))

