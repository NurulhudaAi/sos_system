
#!/usr/bin/env python3
"""
scripts/record_dorm_dataset.py — Interactive Real-time Dataset Recorder & Ground-Truth Labeler
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Designed for testing live CCTV cameras (such as the dormitory RTSP stream):
1. Ingests RTSP video (or webcam) and records high-quality MP4 video.
2. Provides an interactive Web UI at http://localhost:8082 (accessible from mobile phone or PC)
   with 1-click Ground Truth buttons:
   - [🔴 Fall Down (Floor / Mat)]
   - [🛋️ Resting on Bench]
   - [🪑 Sitting on Chair]
   - [🧊 Squatting / Crouching (Fridge/Shoes)]
   - [🚶 Walking (Away / Towards)]
   - [🖐️ Hand SOS (Open -> Thumb -> Fist)]
   - [🙌 Pose SOS (Triangle / V-Shape)]
3. Automatically pairs the recorded video with a structured ground-truth label file:
   - Video: `eval/dorm_dataset/dorm_<timestamp>.mp4`
   - Labels: `eval/dorm_dataset/dorm_<timestamp>_labels.json`
4. Includes automatic verification script that computes Precision, Recall, and False Alarms.
"""

import argparse
import http.server
import json
import logging
import os
import socketserver
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stream_reader import LatestFrameReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("recorder")

DATASET_DIR = ROOT / "eval" / "dorm_dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Shared State
state_lock = threading.Lock()
current_label = "idle"
label_start_time = 0.0
recorded_events: List[Dict[str, Any]] = []
recording_active = True
session_start_time = 0.0
frame_count = 0
latest_jpeg = None


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dorm SOS Dataset Recorder</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: #121316; color: #e5e7eb; padding: 16px; }
    .container { max-width: 900px; margin: 0 auto; }
    h1 { font-size: 20px; color: #38bdf8; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
    .status-bar { background: #1e2025; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; border: 1px solid #2d3139; display: flex; justify-content: space-between; align-items: center; }
    .badge { font-weight: bold; padding: 4px 10px; border-radius: 6px; font-size: 14px; }
    .badge-idle { background: #374151; color: #9ca3af; }
    .badge-active { background: #ef4444; color: white; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
    .video-wrap { position: relative; border-radius: 8px; overflow: hidden; background: #000; margin-bottom: 16px; border: 1px solid #2d3139; }
    .video-wrap img { width: 100%; display: block; height: auto; }
    .button-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 16px; }
    button { padding: 16px; border: none; border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer; transition: all 0.15s ease; display: flex; align-items: center; justify-content: center; gap: 8px; }
    button:active { transform: scale(0.97); }
    .btn-fall { background: #dc2626; color: white; }
    .btn-bench { background: #7c3aed; color: white; }
    .btn-chair { background: #d97706; color: white; }
    .btn-squat { background: #0891b2; color: white; }
    .btn-walk { background: #059669; color: white; }
    .btn-handsos { background: #2563eb; color: white; }
    .btn-posesos { background: #db2777; color: white; }
    .btn-stop { background: #4b5563; color: white; grid-column: 1 / -1; font-size: 17px; padding: 18px; border: 2px solid #9ca3af; }
    .log-box { background: #1e2025; padding: 12px; border-radius: 8px; border: 1px solid #2d3139; max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 13px; }
    .log-entry { margin-bottom: 6px; padding-bottom: 4px; border-bottom: 1px solid #2d3139; color: #94a3b8; }
    .log-entry b { color: #f8fafc; }
  </style>
</head>
<body>
  <div class="container">
    <h1>
      <span>📹 Dorm Dataset Recorder</span>
      <span id="session-time" style="font-size: 14px; color: #94a3b8;">00:00</span>
    </h1>

    <div class="status-bar">
      <div>Current Ground-Truth: <span id="current-label-badge" class="badge badge-idle">IDLE (Normal)</span></div>
      <div id="action-timer" style="color: #cbd5e1; font-family: monospace;">Duration: 0.0s</div>
    </div>

    <div class="video-wrap">
      <img src="/stream" alt="Live Stream">
    </div>

    <div class="button-grid">
      <button class="btn-fall" onclick="setLabel('fall_floor')">🔴 Fall Down (Floor)</button>
      <button class="btn-bench" onclick="setLabel('resting_bench')">🛋️ Resting on Bench</button>
      <button class="btn-chair" onclick="setLabel('sitting_chair')">🪑 Sitting on Chair</button>
      <button class="btn-squat" onclick="setLabel('squatting')">🧊 Squatting (Fridge/Floor)</button>
      <button class="btn-walk" onclick="setLabel('walking_normal')">🚶 Normal Walking</button>
      <button class="btn-handsos" onclick="setLabel('hand_sos')">🖐️ Hand SOS (Help Signal)</button>
      <button class="btn-posesos" onclick="setLabel('pose_sos')">🙌 Pose SOS (Overhead)</button>
      <button class="btn-stop" onclick="setLabel('idle')">⏹️ STOP / NORMAL (Idle)</button>
    </div>

    <h2 style="font-size: 14px; color: #94a3b8; margin-bottom: 8px;">Labeled Actions in this Session:</h2>
    <div id="log-box" class="log-box"></div>
  </div>

  <script>
    let current = "idle";
    let startTime = 0;

    function setLabel(label) {
      fetch("/set_label?label=" + encodeURIComponent(label))
        .then(r => r.json())
        .then(d => {
          updateUI(d);
        });
    }

    function updateUI(d) {
      current = d.current_label;
      startTime = d.label_start;
      const b = document.getElementById("current-label-badge");
      if (current === "idle") {
        b.className = "badge badge-idle";
        b.innerText = "IDLE (Normal)";
      } else {
        b.className = "badge badge-active";
        b.innerText = current.toUpperCase();
      }

      const log = document.getElementById("log-box");
      log.innerHTML = d.events.slice().reverse().map(e => `
        <div class="log-entry">
          <b>${e.label.toUpperCase()}</b>: ${e.start.toFixed(1)}s &rarr; ${e.end.toFixed(1)}s (${e.duration.toFixed(1)}s)
        </div>
      `).join("");
    }

    setInterval(() => {
      fetch("/status").then(r => r.json()).then(d => {
        updateUI(d);
        document.getElementById("session-time").innerText = "Session: " + d.elapsed.toFixed(1) + "s";
        if (d.current_label !== "idle") {
          const actDur = (d.elapsed - d.label_start).toFixed(1);
          document.getElementById("action-timer").innerText = "Duration: " + actDur + "s";
        } else {
          document.getElementById("action-timer").innerText = "Duration: 0.0s";
        }
      });
    }, 500);
  </script>
</body>
</html>
"""


class WebHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        global current_label, label_start_time, recorded_events, session_start_time, latest_jpeg
        now_sec = time.time() - session_start_time

        if self.path == "/" or self.path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif self.path.startswith("/set_label"):
            import urllib.parse
            parsed = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(parsed.query)
            new_label = q.get("label", ["idle"])[0]

            with state_lock:
                if current_label != "idle":
                    # Close previous label event
                    dur = now_sec - label_start_time
                    if dur >= 0.5:
                        recorded_events.append({
                            "label": current_label,
                            "start": round(label_start_time, 2),
                            "end": round(now_sec, 2),
                            "duration": round(dur, 2),
                        })
                current_label = new_label
                label_start_time = now_sec

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "current_label": current_label,
                "label_start": label_start_time,
                "events": recorded_events,
                "elapsed": now_sec,
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "current_label": current_label,
                "label_start": label_start_time,
                "events": recorded_events,
                "elapsed": now_sec,
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while recording_active:
                with state_lock:
                    frame_bytes = latest_jpeg
                if frame_bytes is not None:
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                    except Exception:
                        break
                time.sleep(0.04)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress request log spam


def start_recorder(source: str, port: int = 8082, max_minutes: int = 60):
    global recording_active, session_start_time, latest_jpeg, frame_count, current_label, label_start_time

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_video_path = DATASET_DIR / f"dorm_{timestamp_str}.mp4"
    out_label_path = DATASET_DIR / f"dorm_{timestamp_str}_labels.json"

    # Start HTTP Web UI
    server = socketserver.ThreadingTCPServer(("0.0.0.0", port), WebHandler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    logger.info(f"🌐 Dataset Recorder Web UI running at http://localhost:{port}")
    logger.info(f"📁 Video will be saved to: {out_video_path}")
    logger.info(f"📋 Labels will be saved to: {out_label_path}")

    # Connect to Video Source
    is_rtsp = source.startswith("rtsp://")
    reader = None
    cap = None
    fps = 25.0
    w, h = 1920, 1080

    if is_rtsp:
        logger.info(f"Connecting to RTSP Stream: {source[:45]}...")
        reader = LatestFrameReader(source, name="dorm_recorder")
        if not reader.start():
            logger.error(f"❌ Failed to start stream from {source}")
            return
        # Fetch one sample frame to detect geometry
        for _ in range(30):
            ret, frame, _ = reader.read(timeout=2.0)
            if ret and frame is not None:
                h, w = frame.shape[:2]
                break
            time.sleep(0.1)
    else:
        src = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            logger.error(f"❌ Failed to open video source: {src}")
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    # Prepare VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))

    session_start_time = time.time()
    logger.info(f"🎬 RECORDING STARTED! (Resolution: {w}x{h} @ {fps:.1f} FPS)")
    print("\n" + "═" * 70)
    print(f"👉 OPEN YOUR BROWSER AT: http://localhost:{port}")
    print("   Tap the action buttons as you perform each movement in the room.")
    print("   Press Ctrl+C in this terminal when finished.")
    print("═" * 70 + "\n")

    try:
        while True:
            elapsed = time.time() - session_start_time
            if elapsed >= max_minutes * 60:
                logger.info(f"Reached max duration of {max_minutes} minutes. Finishing recording.")
                break

            if is_rtsp:
                ret, frame, _ = reader.read(timeout=3.0)
            else:
                ret, frame = cap.read()

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            writer.write(frame)

            # Create web stream preview overlay
            vis = frame.copy()
            cur_l = current_label
            badge_color = (0, 0, 255) if cur_l != "idle" else (100, 100, 100)
            tag = f"REC: {elapsed:.1f}s | LABEL: [{cur_l.upper()}]"
            cv2.rectangle(vis, (10, 10), (10 + len(tag) * 14, 50), (0, 0, 0), -1)
            cv2.putText(vis, tag, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, badge_color, 2)

            _, enc = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with state_lock:
                latest_jpeg = enc.tobytes()

    except KeyboardInterrupt:
        logger.info("\n⏹️ Stopping recording on user interrupt (Ctrl+C)...")
    finally:
        recording_active = False
        final_elapsed = time.time() - session_start_time
        with state_lock:
            if current_label != "idle":
                dur = final_elapsed - label_start_time
                if dur >= 0.5:
                    recorded_events.append({
                        "label": current_label,
                        "start": round(label_start_time, 2),
                        "end": round(final_elapsed, 2),
                        "duration": round(dur, 2),
                    })

        writer.release()
        if reader is not None:
            reader.release()
        if cap is not None:
            cap.release()

        # Save Label File
        dataset_meta = {
            "session_id": f"dorm_{timestamp_str}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "duration_seconds": round(final_elapsed, 2),
            "frame_count": frame_count,
            "fps": round(frame_count / max(1e-3, final_elapsed), 2),
            "resolution": [w, h],
            "video_file": str(out_video_path.name),
            "events": recorded_events,
        }
        out_label_path.write_text(json.dumps(dataset_meta, indent=2))
        logger.info(f"✅ Video saved: {out_video_path} ({frame_count} frames, {final_elapsed:.1f}s)")
        logger.info(f"✅ Ground truth labels saved: {out_label_path} ({len(recorded_events)} events)")
        print("\n" + "═" * 70)
        print(f"🎉 Recording Complete!")
        print(f"   Video:  {out_video_path}")
        print(f"   Labels: {out_label_path}")
        print(f"   Run evaluation with:")
        print(f"   python3 eval/evaluate_dorm_dataset.py --video \"{out_video_path}\"")
        print("═" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record live camera session with real-time ground truth labels")
    parser.add_argument("--source", "-s", type=str,
                        default="rtsp://mfustream:mediamfu2025@172.28.107.86:554/cam/realmonitor?channel=1&subtype=0",
                        help="Camera RTSP stream or device index")
    parser.add_argument("--port", "-p", type=int, default=8082, help="Web UI port (default 8082)")
    parser.add_argument("--max-minutes", "-m", type=int, default=60, help="Maximum recording duration in minutes")
    args = parser.parse_args()

    start_recorder(source=args.source, port=args.port, max_minutes=args.max_minutes)
