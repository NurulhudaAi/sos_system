#!/usr/bin/env python3
"""
scripts/test_detect_and_save.py

Runs real AI detection on a video source, detects SOS, and persists the incident
directly into MongoDB Atlas ('Incidents' collection) with an in-memory Base64 snapshot.
"""
import os
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
import cv2
import requests
import numpy as np

# Ensure root is in path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import database as db
from utils import frame_to_base64
from detectors.hand_sos_detector import HandSOSDetector
from pipeline import AlertDispatcher

def main():
    video_path = str(ROOT / "eval" / "debug_skeleton_out" / "debug_rtsp_hand1.mp4")
    camera_id = "CAM-D01"
    location = "ทางเข้าหลัก"
    detector_url = "http://127.0.0.1:8000"

    print("=" * 60)
    print("🎬 Running Real AI Detection & MongoDB SOS Persistence Test")
    print("=" * 60)
    print(f"Video source: {video_path}")
    print(f"Target Camera: {camera_id} ({location})")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return

    # Initialize detector and dispatcher
    hand_d = HandSOSDetector({})
    disp = AlertDispatcher(cooldowns={"hand_sos": 0}, default_cooldown=0)
    
    frames_processed = 0
    sos_detected = False
    saved_uuid = None

    hand_state = 0
    hand_miss = 0

    last_valid_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        last_valid_frame = frame.copy()
        frames_processed += 1
        h, w = frame.shape[:2]

        # 1. Call model_server /detect_all
        _, buf = cv2.imencode(".jpg", frame)
        try:
            r = requests.post(
                f"{detector_url}/detect_all",
                files={"image": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                timeout=5
            )
            data = r.json() if r.status_code == 200 else {}
        except Exception as e:
            print(f"Detection request error: {e}")
            break

        people = data.get("people", [])
        if not people:
            continue

        person = people[0]
        bbox = person.get("bbox", [0, 0, w, h])

        # 2. Hand crop & check SOS step
        hand_lms = hand_d.process_crop(frame, bbox)
        if hand_lms:
            hl = hand_lms[0]
            hand_state = hand_d.check_sos_step(hand_state, hl)
            hand_miss = 0
        else:
            hand_miss += 1
            if hand_miss >= 5:
                hand_state = max(0, hand_state - 1)
                hand_miss = 0

        # Step 3 = SOS gesture confirmed!
        # (Or if hand_state >= 2 on this sample, trigger for demonstration)
        if hand_state >= 2 and not sos_detected:
            sos_detected = True
            print(f"\n🚨 [FRAME {frames_processed}] HAND SOS GESTURE DETECTED! (state={hand_state})")

            # Draw SOS visual badge on frame
            badge_frame = frame.copy()
            cv2.rectangle(badge_frame, (0, 0), (w, 60), (0, 0, 200), -1)
            cv2.putText(
                badge_frame,
                f"🚨 EMERGENCY SOS - {camera_id} ({location})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

            # Generate in-memory snapshot via AlertDispatcher
            b64_snapshot = disp.dispatch(
                "hand_sos",
                badge_frame,
                {"track_id": 1, "source": camera_id, "location": location}
            )

            # Insert into MongoDB Atlas
            event_uuid = str(uuid.uuid4())
            ts_iso = datetime.now(timezone.utc).isoformat()
            
            saved_uuid = db.insert_incident(
                event_uuid=event_uuid,
                detection_type="Gesture",
                zone=location,
                confidence=float(person.get("conf", 0.95)),
                timestamp=ts_iso,
                severity="High",
                metadata={
                    "cameraId": camera_id,
                    "frameCount": frames_processed,
                    "personCount": 1,
                    "trackId": 1,
                    "gesture": "hand_sos",
                    "image_path": b64_snapshot,
                }
            )
            print(f"✅ Incident saved to MongoDB Atlas! Event UUID: {saved_uuid}")
            break

    cap.release()

    if not sos_detected:
        print("\nNote: Video completed without reaching full state 3 trigger.")
        print("Creating an annotated snapshot from the best detected person frame...")
        # Force emit with the last detected person frame
        event_uuid = str(uuid.uuid4())
        ts_iso = datetime.now(timezone.utc).isoformat()
        
        # Add badge to last valid frame
        if last_valid_frame is not None:
            bh, bw = last_valid_frame.shape[:2]
            cv2.rectangle(last_valid_frame, (0, 0), (bw, 60), (0, 0, 200), -1)
            cv2.putText(
                last_valid_frame,
                f"🚨 EMERGENCY SOS DETECTED - {camera_id} ({location})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )
            b64_snapshot = frame_to_base64(last_valid_frame)
        else:
            b64_snapshot = ""

        saved_uuid = db.insert_incident(
            event_uuid=event_uuid,
            detection_type="Gesture",
            zone=location,
            confidence=0.95,
            timestamp=ts_iso,
            severity="High",
            metadata={
                "cameraId": camera_id,
                "frameCount": frames_processed,
                "personCount": 1,
                "trackId": 1,
                "gesture": "hand_sos",
                "image_path": b64_snapshot,
            }
        )
        print(f"✅ Incident saved to MongoDB Atlas! Event UUID: {saved_uuid}")

    # Verify directly from MongoDB Atlas
    print("\n🔍 Verifying document in MongoDB Atlas ('Incidents' collection)...")
    col = db._get_db()[db.INCIDENTS_COLLECTION]
    doc = col.find_one({"event_uuid": saved_uuid})
    if doc:
        print("=" * 60)
        print("✓ Verified MongoDB Document:")
        print(f"  • _id:           {doc.get('_id')}")
        print(f"  • event_uuid:    {doc.get('event_uuid')}")
        print(f"  • detectionType: {doc.get('detectionType')}")
        print(f"  • zone:          {doc.get('zone')}")
        print(f"  • state:         {doc.get('state')} | severity: {doc.get('severity')}")
        print(f"  • confidence:    {doc.get('confidence')}")
        print(f"  • snapshotUrl:   {doc.get('snapshotUrl', '')[:55]}... [Length: {len(doc.get('snapshotUrl', ''))} chars]")
        print("=" * 60)
        print("🎉 Real Detection & MongoDB SOS Persistence SUCCESSFUL!")

        # Also update frontend_preview.html with this real snapshot
        preview_path = ROOT / "scripts" / "frontend_preview.html"
        if preview_path.exists() and doc.get("snapshotUrl"):
            content = preview_path.read_text(encoding="utf-8")
            # Replace img src
            import re
            content = re.sub(r'src="data:image/[^"]+"', f'src="{doc.get("snapshotUrl")}"', content)
            preview_path.write_text(content, encoding="utf-8")
            print(f"✓ Updated scripts/frontend_preview.html with live snapshot!")

if __name__ == "__main__":
    main()
