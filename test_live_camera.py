#!/usr/bin/env python3
"""
test_live_camera.py — Real-time RTSP & Webcam Test Runner with Live Web View
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features:
  1. Live Camera / RTSP ingestion via LatestFrameReader (zero-lag, no buffering).
  2. Full Real-time Detection:
     - Fall Detection: 4s confirm -> 5s recovery -> 10s alert delay (cooldown 60s)
     - Action Plan Auto-Recovery: detects when subject stands back up.
     - Gesture SOS (Pose): Option 1 (overhead triangle) & Option 2 (overhead V-shape) 7-10s (cooldown 30s).
     - Gesture SOS (Hand): Canadian 3-step Hand Signal for Help (Open -> Thumb-in -> Fist) (cooldown 30s).
  3. Built-in Web Server at http://localhost:8081:
     - Responsive Dark UI showing the live annotated stream, status, and alerts.
  4. Automatic MongoDB Atlas dispatch (--send-db) with Base64 snapshots.
"""

import argparse
import http.server
import io
import json
import logging
import math
import os
import socketserver
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detectors.fall_detector import FallDetector
from detectors.hands_over_head_detector import HandsOverHeadDetector
from detectors.hand_sos_detector import HandSOSDetector
from tracker import HungarianTracker
from stream_reader import LatestFrameReader
from utils import preprocess, create_bbox_snapshot_base64
from pipeline import ZoneManager
import database as db_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_tester")

# ── Global Shared Frame for HTTP Streaming ──
latest_annotated_jpeg = None
frame_lock = threading.Lock()
recent_alerts = deque(maxlen=10)

SKELETON_PAIRS = [
    (5, 6),   # shoulders
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    (5, 11), (6, 12),  # torso
    (11, 12),          # hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16), # right leg
]


def _are_both_arms_over_head(kps, conf_thresh=0.25):
    """Option 2: Wide V-shape both arms raised overhead."""
    if len(kps) < 11:
        return False
    l_wr, r_wr = kps[9], kps[10]
    nose = kps[0]
    l_sh, r_sh = kps[5], kps[6]
    if float(l_wr[2]) < conf_thresh or float(r_wr[2]) < conf_thresh:
        return False
    if float(l_sh[2]) < conf_thresh or float(r_sh[2]) < conf_thresh:
        return False

    shoulder_y = min(float(l_sh[1]), float(r_sh[1]))
    shoulder_w = abs(float(l_sh[0]) - float(r_sh[0]))
    wrist_sep = abs(float(l_wr[0]) - float(r_wr[0]))

    min_sep = max(18.0, 0.55 * shoulder_w)
    if wrist_sep < min_sep:
        return False

    head_ref = float(nose[1]) if float(nose[2]) >= conf_thresh else (shoulder_y - 15.0)
    return (
        float(l_wr[1]) <= head_ref + 5.0
        and float(r_wr[1]) <= head_ref + 5.0
        and float(l_wr[1]) <= shoulder_y - 12.0
        and float(r_wr[1]) <= shoulder_y - 12.0
    )


def api_detect_all(frame, server_url="http://127.0.0.1:8000"):
    try:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        r = requests.post(
            f"{server_url}/detect_all",
            files={"image": ("frame.jpg", buf.tobytes(), "image/jpeg")},
            timeout=3.0,
        )
        if r.status_code == 200:
            return r.json()
        else:
            logger.warning(f"Detection API status {r.status_code}: {r.text}")
    except Exception as e:
        logger.debug(f"API detect error: {e}")
    return {}


def nms_containment(people, iou_thresh=0.75, containment_thresh=0.85):
    if len(people) <= 1:
        return people
    boxes = [p["bbox"] for p in people]
    keep = []
    for i, b1 in enumerate(boxes):
        suppress = False
        area1 = max(1.0, (b1[2]-b1[0]) * (b1[3]-b1[1]))
        for j, b2 in enumerate(boxes):
            if i == j: continue
            area2 = max(1.0, (b2[2]-b2[0]) * (b2[3]-b2[1]))
            ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter > 0:
                if (inter / area1) >= containment_thresh and area1 < area2:
                    suppress = True; break
                iou = inter / (area1 + area2 - inter)
                if iou >= iou_thresh and people[i].get("conf", 0) < people[j].get("conf", 0):
                    suppress = True; break
        if not suppress:
            keep.append(people[i])
    return keep


class MJPEGHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_annotated_jpeg, recent_alerts
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Live CCTV SOS System — Real-Time Monitor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0b0f19;
      color: #f8fafc;
      margin: 0;
      padding: 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    .header {{
      text-align: center;
      margin-bottom: 20px;
    }}
    h1 {{
      font-size: 24px;
      margin: 0 0 8px 0;
      color: #38bdf8;
    }}
    .status-badge {{
      display: inline-block;
      padding: 6px 16px;
      border-radius: 9999px;
      background: #1e293b;
      border: 1px solid #334155;
      font-size: 14px;
      font-weight: 600;
      color: #10b981;
    }}
    .video-container {{
      position: relative;
      background: #000;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
      border: 1px solid #334155;
      max-width: 1100px;
      width: 100%;
    }}
    .video-feed {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      max-width: 1100px;
      width: 100%;
      margin-top: 20px;
    }}
    .card {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 10px;
      padding: 16px;
    }}
    h3 {{
      margin-top: 0;
      font-size: 16px;
      color: #94a3b8;
      border-bottom: 1px solid #334155;
      padding-bottom: 8px;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
      font-size: 14px;
      line-height: 1.6;
    }}
    .alert-item {{
      background: #0f172a;
      border-left: 4px solid #ef4444;
      padding: 8px 12px;
      margin-bottom: 8px;
      border-radius: 4px;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>🚨 SOS AI Detection — Live CCTV Monitor</h1>
    <div class="status-badge">● LIVE STREAM ACTIVE (Port 8081)</div>
  </div>
  
  <div class="video-container">
    <img class="video-feed" src="/video_feed" alt="Live Stream" />
  </div>

  <div class="card-grid">
    <div class="card">
      <h3>📋 วิธีทดสอบสัญญาณขอความช่วยเหลือ (SOS Testing)</h3>
      <ul>
        <li><strong>Gesture SOS (Hand Signal):</strong> 3 จังหวะหน้ากล้อง (1. แบมือ &rarr; 2. พับนิ้วโป้ง &rarr; 3. กำหมัดทับนิ้วโป้ง)</li>
        <li><strong>Gesture SOS (Option 1):</strong> สองมือประกบเป็นสามเหลี่ยมเหนือศีรษะค้างไว้ 7–10 วิ (Cooldown 30s)</li>
        <li><strong>Gesture SOS (Option 2):</strong> ชูสองมือขึ้นเหนือหัวเป็นรูปตัว V ค้างไว้ 7–10 วิ (Cooldown 30s)</li>
        <li><strong>Fall Detection:</strong> นอนราบบนพื้น (4s ยืนยันล้ม &rarr; รอ 5s ฟื้นตัว &rarr; ส่งแจ้งเตือนที่ 10s)</li>
        <li><strong>Auto-Recovery:</strong> เมื่อลุกขึ้นยืน ระบบจะตรวจจับและบันทึกว่าฟื้นตัวแล้วทันที</li>
      </ul>
    </div>
    <div class="card">
      <h3>🔔 ประวัติการแจ้งเตือนล่าสุด (Recent Events)</h3>
      <div id="alerts">
        {"".join(f'<div class="alert-item">⏱ {a["time"]} — <strong>{a["type"]}</strong>: {a["label"]}</div>' for a in recent_alerts) if recent_alerts else '<p style="color:#64748b;font-size:14px;">ยังไม่มีเหตุการณ์แจ้งเตือน</p>'}
      </div>
    </div>
  </div>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/video_feed":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        jpg = latest_annotated_jpeg
                    if jpg is not None:
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpg)))
                        self.end_headers()
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.04)  # ~25 FPS
            except Exception:
                pass
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # silence http log


def start_web_server(port=8081):
    class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    try:
        server = ThreadedHTTPServer(("", port), MJPEGHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        logger.info(f"🌐 Web Stream Server is running at: http://localhost:{port}")
        return server
    except Exception as e:
        logger.error(f"Failed to start web server on port {port}: {e}")
        return None


def run_live_stream(source, port=8081, send_db=True, cam_id="default"):
    global latest_annotated_jpeg, recent_alerts
    start_web_server(port=port)

    # Thresholds & Configs
    cfg = yaml.safe_load((ROOT / "config/thresholds.yaml").read_text())
    fall_cfg = cfg.get("fall", {})
    hand_cfg = cfg.get("hand_sos", {})
    pose_cfg = cfg.get("hands_over_head", {})
    # Zero-config Zone Manager: defaults to Full-Frame detection everywhere
    zones = ZoneManager(cam_id=cam_id)
    logger.info(f"Zones initialized for cam_id={cam_id!r}: {len(zones.zones)} active (send_db={send_db})")

    tracker = HungarianTracker(frame_w=1920, frame_h=1080, max_lost=30)
    fall_d = FallDetector(fall_cfg)
    head_d = HandsOverHeadDetector(pose_cfg)
    hand_d = HandSOSDetector(hand_cfg)

    # Timers & States
    fall_cooldown = float(fall_cfg.get("cooldown_seconds", 60))
    fall_dispatch_delay = float(fall_cfg.get("dispatch_delay_seconds", 10.0))
    pose_cooldown = float(pose_cfg.get("cooldown_seconds", 30))
    hand_cooldown = float(hand_cfg.get("cooldown_seconds", 30))

    fall_first_down_ts = {}
    active_fall_alerts = {}
    fall_stand_count = {}
    fall_ev = {}
    _fall_last_ts = {}

    overhead_start_ts = {}
    _pose_last_ts = {}

    # Hand SOS States
    hand_states = {}
    hand_step_time = {}
    hand_miss = {}
    hand_consec3 = {}
    hand_recent_sos = defaultdict(lambda: deque(maxlen=10))
    _hand_last_ts = {}

    # Open Capture Source using LatestFrameReader if RTSP, or VideoCapture if local device
    is_rtsp = isinstance(source, str) and source.startswith("rtsp://")
    reader = None
    cap = None

    if is_rtsp:
        logger.info(f"Opening RTSP stream with LatestFrameReader: {source[:40]}...")
        reader = LatestFrameReader(source, name="live_stream")
        if not reader.start():
            logger.error(f"❌ Failed to start LatestFrameReader for {source}")
            return
    else:
        src = int(source) if str(source).isdigit() else str(source)
        logger.info(f"Opening local capture device: {src}")
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            logger.error(f"❌ Failed to open video source: {src}")
            return
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    logger.info(f"✅ Camera Connected! Open your browser to view: http://localhost:{port}")

    t_start = time.time()
    frame_idx = 0

    while True:
        if reader is not None:
            ret, raw, cap_ts = reader.read(timeout=5.0)
        else:
            ret, raw = cap.read()
            cap_ts = time.time()

        if not ret or raw is None:
            time.sleep(0.05)
            continue

        frame_idx += 1
        now = time.time()
        elapsed_sec = now - t_start

        frame = preprocess(raw, 1920, 1080)
        h, w = 1080, 1920

        # Inference on each frame
        resp = api_detect_all(frame)
        people = resp.get("people", [])
        assigned = tracker.update(nms_containment(people))

        vis = frame.copy()
        cv2.putText(vis, f"TIME: {elapsed_sec:.1f}s | DETECTED: {len(assigned)} | Port: {port}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        for p in assigned:
            tid = p.get("track_id")
            bbox = [int(v) for v in p.get("bbox", [0, 0, 0, 0])]
            conf = float(p.get("conf", 0.0))
            kps = np.array(p.get("keypoints", []))
            posture_class = p.get("posture_class")
            posture_conf = float(p.get("posture_conf", 0.0))

            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h

            # Process Fall
            fr = fall_d.process(tid, kps, bbox, h, w, posture_class=posture_class, posture_conf=posture_conf, timestamp=elapsed_sec,
                                on_furniture=bool(zones.furniture_at(kps, bbox, w, h)))
            is_critical = fr.get("is_critical") and not fr.get("recovered_quickly")
            is_confirmed = (fr.get("is_fallen") or fr.get("danger_lying")) and not fr.get("recovered_quickly")
            is_down = fr.get("is_down", False)

            # Auto-Recovery
            if tid in active_fall_alerts:
                af = active_fall_alerts[tid]
                is_standing_now = (not is_down and not fr.get("is_fallen") and (fr.get("spine_angle", 0) >= 50.0 or fr.get("is_sitting") or fr.get("is_crouching")))
                if is_standing_now:
                    fall_stand_count[tid] = fall_stand_count.get(tid, 0) + 1
                    if fall_stand_count[tid] >= 3 and not af.get("recovered"):
                        af["recovered"] = True
                        logger.info(f"🌟 [ACTION PLAN] Person [tid={tid}] stood up & recovered at {elapsed_sec:.1f}s!")
                        if send_db and af.get("uuid"):
                            try:
                                db_module.resolve_incident(
                                    af["uuid"],
                                    resolved_by="auto_recovery",
                                    notes=f"Subject recovered and stood up at t={elapsed_sec:.1f}s",
                                    post_event_status="RECOVERED_STAND_UP"
                                )
                            except Exception:
                                pass
                else:
                    fall_stand_count[tid] = 0

            # Fall Lifecycle Timer
            if is_confirmed or is_critical:
                if tid not in fall_first_down_ts:
                    fall_first_down_ts[tid] = fr.get("ground_time") or (elapsed_sec - 4.0)

                down_duration = elapsed_sec - fall_first_down_ts[tid]
                if down_duration >= fall_dispatch_delay or is_critical:
                    prev_f = _fall_last_ts.get(tid, -1e9)
                    if (elapsed_sec - prev_f) >= fall_cooldown and not fall_ev.get(tid):
                        event_conf = fr.get("event_confidence") or 0.90
                        lbl = f"FALL [tid={tid}] down={down_duration:.1f}s"
                        logger.info(f"🚨 [ALERT DISPATCHED] {lbl} (Event Conf: {event_conf:.2f})")
                        recent_alerts.appendleft({"time": f"{elapsed_sec:.1f}s", "type": "Fall", "label": lbl})
                        _fall_last_ts[tid] = elapsed_sec
                        fall_ev[tid] = True
                        ev_uuid = ""
                        if send_db:
                            try:
                                b64 = create_bbox_snapshot_base64(frame, bbox=bbox, label=lbl, color=(0, 0, 255))
                                ev_uuid = db_module.insert_incident(
                                    detection_type="Fall",
                                    zone="Live_Camera",
                                    confidence=event_conf,
                                    snapshotUrl=b64,
                                    metadata={"track_id": tid, "time_down": round(down_duration, 1)}
                                )
                            except Exception as e:
                                logger.error(f"DB insert error: {e}")
                        active_fall_alerts[tid] = {"uuid": ev_uuid, "alert_ts": elapsed_sec, "recovered": False}
            else:
                is_recovery_stand = bool(
                    fr.get("recovered_quickly")
                    or fr.get("is_sitting")
                    or fr.get("is_crouching")
                    or (not is_down and not fr.get("is_fallen") and fr.get("spine_angle", 0) >= 48.0)
                )
                if is_recovery_stand and tid in fall_first_down_ts:
                    logger.info(f"ℹ️ Fall recovery detected within window (tid={tid}) — Alert suppressed.")
                    fall_first_down_ts.pop(tid, None)
                if fr.get("is_sitting") or fr.get("is_crouching") or not fr.get("is_fallen"):
                    fall_ev[tid] = False

            # Gesture SOS (Pose: Option 1 Triangle / Option 2 Overhead V)
            is_lying = fr.get("is_fallen") or (fr.get("spine_angle") and fr.get("spine_angle") < 40.0)
            both_overhead = _are_both_arms_over_head(kps)
            head_res = head_d.process(tid, kps, h, w, timestamp=elapsed_sec) if not is_lying else {}

            is_pose_active = both_overhead or head_res.get("hands_over_head", False)
            if is_pose_active and not is_lying:
                if tid not in overhead_start_ts:
                    overhead_start_ts[tid] = elapsed_sec
                hold_sec = elapsed_sec - overhead_start_ts[tid]
                pose_type_str = "Overhead Triangle" if head_res.get("hands_over_head") else "Both Hands Overhead"

                if hold_sec >= 7.0:
                    prev_p = _pose_last_ts.get(tid, -1e9)
                    if (elapsed_sec - prev_p) >= pose_cooldown:
                        lbl = f"GESTURE SOS [tid={tid}] {pose_type_str} held={hold_sec:.1f}s"
                        logger.info(f"🙌 [ALERT DISPATCHED] {lbl}")
                        recent_alerts.appendleft({"time": f"{elapsed_sec:.1f}s", "type": "Gesture", "label": lbl})
                        _pose_last_ts[tid] = elapsed_sec
                        if send_db:
                            try:
                                b64 = create_bbox_snapshot_base64(frame, bbox=bbox, label=lbl, color=(0, 165, 255))
                                db_module.insert_incident(
                                    detection_type="Gesture",
                                    zone="Live_Camera",
                                    confidence=0.98,
                                    snapshotUrl=b64,
                                    metadata={"track_id": tid, "gesture_type": pose_type_str, "hold_sec": round(hold_sec, 1)}
                                )
                            except Exception as e:
                                logger.error(f"DB insert error: {e}")
            else:
                overhead_start_ts.pop(tid, None)

            # Gesture SOS (Hand Signal for Help: Palm -> Thumb-in -> Fist)
            # Strictly requires person facing camera with elevated hand (not walking with arms down)
            hand_badge = ""
            face_visible = False
            if len(kps) >= 3:
                face_visible = any(float(kps[i][2]) >= 0.15 for i in (0, 1, 2))

            if not both_overhead and not is_lying and not fall_ev.get(tid) and face_visible:
                prev_hs = hand_states.get(tid, 0)
                hs = prev_hs
                hdet = False
                try:
                    hand_lms = hand_d.process_wrist_crop(frame, kps, h, w, bbox=bbox)
                    if hand_lms:
                        hl = hand_lms[0]
                        hdet = True
                        hs = hand_d.check_sos_step(hs, hl)
                except Exception:
                    pass

                if hs != prev_hs:
                    hand_step_time[tid] = elapsed_sec
                elif hs >= 1 and (elapsed_sec - hand_step_time.get(tid, elapsed_sec)) > 3.5:
                    hs = 0
                    hand_consec3[tid] = 0

                if hdet:
                    hand_miss[tid] = 0
                else:
                    hand_miss[tid] = hand_miss.get(tid, 0) + 1
                    if hand_miss[tid] >= 8:
                        hs = max(0, hs - 1)
                        hand_miss[tid] = 0

                hand_states[tid] = hs

                # Consecutive Fist check
                is_fist = bool(hdet and hs == 3 and not hand_d._palm_open(hl) and hand_d._fingers_closed(hl))
                if is_fist:
                    hand_consec3[tid] = hand_consec3.get(tid, 0) + 1
                elif hdet:
                    hand_consec3[tid] = 0

                consec_ok = hand_consec3.get(tid, 0) >= 2
                is_sos = bool(hs == 3 and consec_ok)
                hand_recent_sos[tid].append(is_sos)
                sos_hits = sum(1 for ok in hand_recent_sos[tid] if ok)

                # Show hand step on screen
                step_names = {1: "Open Palm", 2: "Thumb In", 3: "SOS Fist!"}
                if hs in step_names:
                    hand_badge = f"Hand: [{hs}/3] {step_names[hs]}"

                if sos_hits >= 2:
                    prev_h = _hand_last_ts.get(tid, -1e9)
                    if (elapsed_sec - prev_h) >= hand_cooldown:
                        lbl = f"GESTURE SOS [tid={tid}] Hand Signal for Help"
                        logger.info(f"[ALERT DISPATCHED] {lbl}")
                        recent_alerts.appendleft({"time": f"{elapsed_sec:.1f}s", "type": "Gesture", "label": lbl})
                        _hand_last_ts[tid] = elapsed_sec
                        if send_db:
                            try:
                                b64 = create_bbox_snapshot_base64(frame, bbox=bbox, label=lbl, color=(0, 200, 0))
                                db_module.insert_incident(
                                    detection_type="Gesture",
                                    zone="Live_Camera",
                                    confidence=0.95,
                                    snapshotUrl=b64,
                                    metadata={"track_id": tid, "gesture_type": "Hand SOS Signal", "held_sec": 3.0}
                                )
                            except Exception as e:
                                logger.error(f"DB insert error: {e}")
                        # Reset for next time
                        hand_states[tid] = 0
                        hand_consec3[tid] = 0
                        hand_recent_sos[tid].clear()
            else:
                # Reset hand tracking if back to camera, lying, or in other alert
                hand_states[tid] = 0
                hand_consec3[tid] = 0
                hand_recent_sos[tid].clear()

            # Draw Overlays on Frame
            has_fall_evidence = fr.get("has_fall_evidence", False)
            box_color = (0, 255, 0)
            status_tag = f"ID:{tid} [NORMAL] ({fr.get('spine_angle', 0):.0f}deg)"

            if is_confirmed or fr.get("is_fallen"):
                box_color = (0, 0, 255)
                down_sec = elapsed_sec - fall_first_down_ts.get(tid, elapsed_sec)
                status_tag = f"ID:{tid} [FALL ALERT] down={down_sec:.1f}s"
            elif tid in fall_first_down_ts and is_down and has_fall_evidence:
                box_color = (0, 140, 255)  # Orange warning
                down_sec = elapsed_sec - fall_first_down_ts[tid]
                status_tag = f"ID:{tid} [FALL PENDING: {down_sec:.1f}s/4.0s]"
            elif fr.get("on_furniture") and not fr.get("is_sitting"):
                box_color = (200, 120, 255)
                status_tag = f"ID:{tid} [ON FURNITURE / RESTING]"
            elif is_down and not has_fall_evidence:
                # Static resting on furniture (bench/couch/bed) — NOT a fall!
                box_color = (180, 180, 180)  # Neutral gray
                status_tag = f"ID:{tid} [RESTING / BENCH] ({fr.get('spine_angle', 0):.0f}deg)"
            elif is_pose_active:
                box_color = (0, 165, 255)
                held = elapsed_sec - overhead_start_ts.get(tid, elapsed_sec)
                status_tag = f"ID:{tid} [GESTURE SOS: {held:.1f}s/7.0s]"
            elif hand_badge:
                box_color = (0, 220, 255)
                status_tag = f"ID:{tid} [{hand_badge}]"
            elif fr.get("is_crouching"):
                box_color = (0, 200, 200)
                status_tag = f"ID:{tid} [CROUCHING] ({fr.get('spine_angle', 0):.0f}deg)"
            elif fr.get("is_sitting"):
                box_color = (255, 200, 0)
                status_tag = f"ID:{tid} [SEATED] ({fr.get('spine_angle', 0):.0f}deg)"

            # Draw Skeleton
            if len(kps) >= 17:
                for pt1_idx, pt2_idx in SKELETON_PAIRS:
                    pt1, pt2 = kps[pt1_idx], kps[pt2_idx]
                    if pt1[2] > 0.25 and pt2[2] > 0.25:
                        cv2.line(vis, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), box_color, 2)
                for pt in kps:
                    if pt[2] > 0.25:
                        cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)

            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
            cv2.rectangle(vis, (x1, y1 - 25), (x1 + len(status_tag) * 11, y1), box_color, -1)
            cv2.putText(vis, status_tag, (x1 + 4, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Encode and Update Web Stream
        _, enc = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with frame_lock:
            latest_annotated_jpeg = enc.tobytes()

        # Small sleep to prevent CPU spin
        time.sleep(0.01)

    if reader is not None:
        reader.release()
    if cap is not None:
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Camera SOS Stream & Monitor")
    parser.add_argument("--source", "-s", type=str, default="0", help="Camera index (e.g. 0) or RTSP URL")
    parser.add_argument("--cam-id", "-c", type=str, default="boonsong_fl2", help="Camera identifier for alerts and incidents")
    parser.add_argument("--port", "-p", type=int, default=8081, help="Web stream port (default 8081)")
    parser.add_argument("--send-db", action="store_true", default=True, help="Send alerts to MongoDB Atlas")
    args = parser.parse_args()

    run_live_stream(args.source, port=args.port, send_db=args.send_db, cam_id=args.cam_id)
