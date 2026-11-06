#!/usr/bin/env python3
"""
scripts/test_7_videos_to_db.py
ทดสอบนำผลการตรวจจับจากวิดีโอทดสอบทั้ง 7 กล้องเข้าสู่ฐานข้อมูล MongoDB Atlas (Collection: Incidents)
พร้อมบันทึกภาพ Snapshot (Base64 + File Path), Meta Tags, พิกัด Bounding Box และระดับความรุนแรง
"""

import sys
import os
import json
import uuid
import base64
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import database

def main():
    print("=" * 78)
    print(" [BENCHMARK] Test 7 Benchmark Videos -> MongoDB Atlas")
    print("=" * 78)

    # 1. เช็กการเชื่อมต่อ DB
    db = database._get_db()
    col_name = database.INCIDENTS_COLLECTION
    print(f"[DB] Database: {database.MONGODB_DB_NAME} | Collection: {col_name}")
    print(f"[STATUS] Connection: {'Online' if database.health_check() else 'Offline'}")

    summary_file = ROOT / "eval" / "scan_after_tuning" / "scan_summary.json"
    if not summary_file.exists():
        print(f"[ERROR] Summary file not found: {summary_file}")
        return

    with open(summary_file, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    inserted_records = []
    batch_tag = f"benchmark_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    for video_name, events in benchmarks.items():
        cam_code = video_name.rsplit(".", 1)[0]
        print(f"\n[VIDEO] {video_name} (Camera {cam_code}) -> {len(events)} events found")

        if not events:
            print("   [Clean Pass] Normal Motion (0 False Alarm)")
            continue

        for ev in events:
            ev_uuid = str(uuid.uuid4())
            det_type = ev.get("event_type", "Fall")
            lbl = ev.get("label", "")
            conf = float(ev.get("confidence", 0.90))
            p_conf = float(ev.get("person_confidence", 0.0))
            t_sec = float(ev.get("timestamp_sec", 0.0))
            tid = ev.get("track_id", 0)
            bbox = ev.get("bbox", [0, 0, 0, 0])
            details = ev.get("details", "")

            # กำหนด Severity
            if "multi-modal" in lbl.lower():
                sev = "Critical"
            elif det_type == "Fall":
                sev = "Critical" if "down=10.0s" in lbl and conf >= 0.90 else "High"
            else:
                sev = "Medium"

            # อ่านและแปลงภาพ Snapshot เป็น Base64
            snap_rel = ev.get("snapshot", "")
            snap_path = ROOT / "eval" / "scan_after_tuning" / snap_rel
            raw_b64 = ""
            b64_url = ""
            if snap_path.exists():
                img = cv2.imread(str(snap_path))
                if img is not None:
                    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    raw_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                    b64_url = f"data:image/jpeg;base64,{raw_b64}"

            iso_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            meta = {
                "cameraId":          cam_code,
                "source_id":         cam_code,
                "video_file":        video_name,
                "track_id":          tid,
                "trackId":           tid,
                "person_confidence": p_conf,
                "snapshotUrl":       b64_url,
                "snapshot_base64":   raw_b64,
                "imagePath":         str(snap_path),
                "event_type":        det_type,
                "label":             lbl,
                "video_timestamp_s": t_sec,
                "bbox":              bbox,
                "details":           details,
                "batch_test":        batch_tag,
            }

            doc_id = database.insert_incident(
                event_uuid     = ev_uuid,
                detection_type = det_type,
                zone           = cam_code,
                confidence     = conf,
                timestamp      = iso_ts,
                severity       = sev,
                metadata       = meta,
            )

            inserted_records.append({
                "uuid": ev_uuid,
                "video": video_name,
                "zone": cam_code,
                "type": det_type,
                "time": t_sec,
                "label": lbl,
                "severity": sev,
                "conf": conf,
            })
            print(f"   [SAVED] [{sev:8s}] {det_type:7s} | {lbl} | UUID: {ev_uuid[:8]}...")

    # 3. สรุปผลการ Query จากฐานข้อมูลจริง
    print("\n" + "=" * 78)
    print(" [SUMMARY] MongoDB Incident Verification Report")
    print("=" * 78)
    total_in_col = db[col_name].count_documents({})
    batch_count = db[col_name].count_documents({"metadata.batch_test": batch_tag})

    print(f"- Total incidents in collection `{col_name}`: {total_in_col}")
    print(f"- Incidents inserted in current batch: {batch_count}\n")

    print(f"{'No.':<4} | {'Camera / Zone':<16} | {'Type':<8} | {'Time (s)':<9} | {'Severity':<9} | {'Status':<8} | {'UUID':<12}")
    print("-" * 78)

    for i, r in enumerate(inserted_records, 1):
        # Query สถานะจริงจาก DB
        db_doc = db[col_name].find_one({"event_uuid": r["uuid"]})
        state = db_doc.get("state", "Open") if db_doc else "Unknown"
        print(f"{i:<4} | {r['zone']:<16} | {r['type']:<8} | {r['time']:<8.1f}s | {r['severity']:<9} | {state:<8} | {r['uuid'][:10]}...")

    print("=" * 78)
    print("[OK] Benchmark database testing completed successfully")

if __name__ == "__main__":
    main()
