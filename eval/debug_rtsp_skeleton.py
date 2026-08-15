#!/usr/bin/env python3
"""
eval/debug_rtsp_skeleton.py — Skeleton Debug Tool for RTSP/Video

แสดง/บันทึก debug overlay:
  - COCO-17 skeleton (body keypoints)
  - MediaPipe 21-point hand landmarks
  - Hand SOS state machine status (0→1→2→3)
  - Raw landmark values (distances, ratios, tier checks)
  - Per-frame CSV log สำหรับ post-analysis

รองรับทั้ง RTSP stream จริง และไฟล์วิดีโอ

Usage:
    # จากวิดีโอไฟล์:
    python3 model_server.py &
    python3 eval/debug_rtsp_skeleton.py \\
        --source eval/videos/rtsp_hand3.mp4 \\
        --detector-url http://127.0.0.1:8000 \\
        --out-dir eval/debug_skeleton_out

    # จาก RTSP stream:
    python3 eval/debug_rtsp_skeleton.py \\
        --source "rtsp://mfustream:mediamfu2025@172.28.106.79/Streaming/Channels/101" \\
        --detector-url http://127.0.0.1:8000 \\
        --out-dir eval/debug_skeleton_out \\
        --max-seconds 60

    # เฉพาะ log (ไม่ต้องการวิดีโอ):
    python3 eval/debug_rtsp_skeleton.py \\
        --source eval/videos/rtsp_hand1.mp4 \\
        --no-video
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detectors.hand_sos_detector import HandSOSDetector


# ═══════════════════════════════════════════════════════════════════════
# COCO-17 Skeleton Drawing
# ═══════════════════════════════════════════════════════════════════════
SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

SKELETON_COLORS = {
    'torso': (0, 255, 200), 'left_arm': (255, 180, 0),
    'right_arm': (0, 180, 255), 'left_leg': (200, 100, 255),
    'right_leg': (100, 255, 100),
}

def _conn_color(a, b):
    if (a, b) in [(5, 6), (5, 11), (6, 12), (11, 12)]:
        return SKELETON_COLORS['torso']
    if a in (5, 7, 9) and b in (5, 7, 9):
        return SKELETON_COLORS['left_arm']
    if a in (6, 8, 10) and b in (6, 8, 10):
        return SKELETON_COLORS['right_arm']
    if a in (11, 13, 15) and b in (11, 13, 15):
        return SKELETON_COLORS['left_leg']
    if a in (12, 14, 16) and b in (12, 14, 16):
        return SKELETON_COLORS['right_leg']
    return (200, 200, 200)


def draw_skeleton(frame, kps, conf_thresh=0.3):
    if not kps or len(kps) < 17:
        return
    for a, b in SKELETON_CONNECTIONS:
        if a >= len(kps) or b >= len(kps):
            continue
        ka, kb = kps[a], kps[b]
        if len(ka) < 3 or len(kb) < 3 or ka[2] < conf_thresh or kb[2] < conf_thresh:
            continue
        cv2.line(frame, (int(ka[0]), int(ka[1])), (int(kb[0]), int(kb[1])),
                 _conn_color(a, b), 2)
    for i, kp in enumerate(kps):
        if len(kp) < 3 or kp[2] < conf_thresh:
            continue
        cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 255, 255), -1)
        cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 0, 0), 1)


# ═══════════════════════════════════════════════════════════════════════
# Hand Landmarks Drawing
# ═══════════════════════════════════════════════════════════════════════
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

def draw_hand_on_frame(frame, hl, crop_offset, crop_size):
    cx1, cy1 = crop_offset
    cw, ch = crop_size
    pts = [(cx1 + int(lm.x * cw), cy1 + int(lm.y * ch)) for lm in hl]
    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 220, 0), 2)
    for i, pt in enumerate(pts):
        color = (0, 0, 255) if i == 4 else (0, 255, 0)
        cv2.circle(frame, pt, 4, color, -1)


# ═══════════════════════════════════════════════════════════════════════
# Debug Overlay
# ═══════════════════════════════════════════════════════════════════════
STATE_LABELS = ["--", "palm", "thumb", "SOS!"]
STATE_COLORS = [(200, 200, 200), (255, 200, 0), (0, 180, 255), (0, 255, 0)]

def draw_bbox_with_state(frame, bbox, tid, conf, hs):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = STATE_COLORS[min(hs, 3)]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{tid} {conf:.0%} [{STATE_LABELS[min(hs, 3)]}]"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def draw_debug_panel(frame, debug_info, hs, x_offset=0, y_offset=0):
    if not debug_info:
        return
    y = y_offset + 16
    lines = [
        f"palm={debug_info.get('palm_open','?')} thumb={debug_info.get('thumb_in','?')} "
        f"fing={debug_info.get('fingers_closed','?')} comb={debug_info.get('combined','?')}",
        f"span={debug_info.get('hand_span',0):.3f} thumb_r={debug_info.get('thumb_ratio',0):.2f} "
        f"idx_r={debug_info.get('thumb_idx_ratio',0):.2f}",
    ]
    fingers = debug_info.get('fingers', {})
    for name, fd in fingers.items():
        lines.append(
            f"  {name}: r={fd.get('ratio',0):.2f} "
            f"t1={fd.get('t1_closed','?')} t2={fd.get('t2_closed','?')} t3={fd.get('t3_closed','?')}"
        )
    panel_h = len(lines) * 16 + 10
    panel_w = 420
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_offset, y_offset),
                  (x_offset + panel_w, y_offset + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    for line in lines:
        cv2.putText(frame, line, (x_offset + 4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 200), 1)
        y += 16


def draw_hud(frame, fps, frame_idx, t_sec, hand_states, n_people):
    h, w = frame.shape[:2]
    panel_h = 80 + len(hand_states) * 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, min(panel_h, h)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    y = 18
    cv2.putText(frame, f"FPS:{fps:.1f}  Frame:{frame_idx}  t={t_sec:.1f}s",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y += 20
    cv2.putText(frame, f"People:{n_people}",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
    y += 20
    for tid in sorted(hand_states.keys()):
        hs = hand_states[tid]
        hl = STATE_LABELS[min(hs, 3)]
        color = STATE_COLORS[min(hs, 3)]
        cv2.putText(frame, f"T{tid}: hand={hl}",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y += 18


# ═══════════════════════════════════════════════════════════════════════
# Tracker + API
# ═══════════════════════════════════════════════════════════════════════
class SimpleTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}

    def _iou(self, a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        aa = max(1e-6, (a[2]-a[0])*(a[3]-a[1]))
        ab = max(1e-6, (b[2]-b[0])*(b[3]-b[1]))
        return inter / (aa+ab-inter) if (aa+ab-inter) > 0 else 0.0

    def update(self, dets):
        used = []
        for det in dets:
            best_id, best_iou = None, 0.0
            for tid, t in self.tracks.items():
                iou = self._iou(det["bbox"], t["bbox"])
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_iou >= 0.3 and best_id not in used:
                det["track_id"] = best_id
                self.tracks[best_id].update(bbox=det["bbox"], lost=0)
                used.append(best_id)
            else:
                tid = self.next_id; self.next_id += 1
                det["track_id"] = tid
                self.tracks[tid] = {"bbox": det["bbox"], "lost": 0}
                used.append(tid)
        for tid in list(self.tracks):
            if tid not in {d["track_id"] for d in dets}:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > 5:
                    del self.tracks[tid]
        return dets


def detect_all(detector_url, frame, timeout=5):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    try:
        r = requests.post(detector_url + "/detect_all",
                          files={"image": ("f.jpg", buf.tobytes(), "image/jpeg")},
                          timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"[detect_all] error: {e}")
    return {"people": [], "objects": []}


def apply_clahe(frame):
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return frame


# ═══════════════════════════════════════════════════════════════════════
# Main Processing
# ═══════════════════════════════════════════════════════════════════════
def run_debug(source, detector_url, hand_cfg, out_dir, sample_fps=5,
              save_video=True, max_seconds=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_rtsp = str(source).startswith("rtsp://") or str(source).startswith("http")
    cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG if is_rtsp else cv2.CAP_ANY)
    if is_rtsp:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    if not cap.isOpened():
        print(f"Cannot open: {source}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    frame_interval = max(1, round(src_fps / sample_fps)) if not is_rtsp else 1

    hand_d = HandSOSDetector(hand_cfg)
    tracker = SimpleTracker()
    hand_states = {}
    hand_miss = {}
    GRACE_FRAMES = 5

    debug_writer = None
    if save_video:
        src_name = Path(source).stem if not is_rtsp else "rtsp_live"
        out_path = out_dir / f"debug_{src_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        debug_writer = cv2.VideoWriter(str(out_path), fourcc,
                                        min(sample_fps, src_fps),
                                        (frame_w, frame_h))
        print(f"Debug video -> {out_path}")

    csv_path = out_dir / f"debug_{Path(source).stem if not is_rtsp else 'rtsp_live'}.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "t_sec", "tid", "state", "hdet",
        "palm_open", "thumb_in", "fingers_closed", "combined",
        "hand_span", "thumb_ratio", "thumb_idx_ratio",
        "idx_r", "idx_t1", "idx_t2", "idx_t3",
        "mid_r", "mid_t1", "mid_t2", "mid_t3",
        "ring_r", "ring_t1", "ring_t2", "ring_t3",
        "pinky_r", "pinky_t1", "pinky_t2", "pinky_t3",
    ])

    print(f"\n{'='*60}")
    print(f"Skeleton Debug Tool")
    print(f"{'='*60}")
    print(f"Source: {source}")
    print(f"Detector: {detector_url}")
    print(f"Sample FPS: {sample_fps}")
    print(f"Save video: {save_video}")
    print(f"Max seconds: {max_seconds or 'unlimited'}")
    print(f"CSV log: {csv_path}")
    print(f"{'='*60}\n")

    frame_idx = 0
    t0 = time.time()
    fps_timer = []
    last_sample_t = 0.0

    try:
        while True:
            ret, raw = cap.read()
            if not ret:
                if is_rtsp:
                    print("Stream read failed -- retrying in 2s...")
                    time.sleep(2)
                    continue
                break
            frame_idx += 1

            if not is_rtsp and frame_idx % frame_interval != 0:
                continue

            if is_rtsp:
                now = time.time()
                if now - last_sample_t < (1.0 / sample_fps):
                    continue
                last_sample_t = now

            t_sec = frame_idx / src_fps if not is_rtsp else (time.time() - t0)

            if max_seconds and t_sec > max_seconds:
                print(f"max_seconds={max_seconds} reached")
                break

            frame = raw.copy()
            h, w = frame.shape[:2]

            now = time.time()
            fps_timer = [t for t in fps_timer if now - t < 1.0]
            fps_timer.append(now)
            current_fps = len(fps_timer)

            resp = detect_all(detector_url, frame)
            people_raw = resp.get("people", [])
            assigned = tracker.update([{"bbox": p["bbox"]} for p in people_raw])

            for i, det in enumerate(assigned):
                if i < len(people_raw):
                    det["keypoints"] = people_raw[i].get("keypoints", [])
                    det["conf"] = people_raw[i].get("conf", 0.0)

            frame_clahe = apply_clahe(frame)

            for det in assigned:
                tid = det["track_id"]
                bbox = det["bbox"]
                kps = det.get("keypoints", [])
                conf = det.get("conf", 0.0)
                hs = hand_states.get(tid, 0)
                hdet = False
                debug_info = {}

                draw_skeleton(frame, kps)

                x1, y1, x2, y2 = [int(v) for v in bbox]
                bw, bh = x2 - x1, y2 - y1
                cx1 = max(0, x1 - int(bw * 0.1))
                cy1 = max(0, y1 - int(bh * 0.1))
                cx2 = min(w, x2 + int(bw * 0.1))
                cy2 = min(h, y2 + int(bh * 0.1))

                try:
                    hand_lms = hand_d.process_crop(frame_clahe, bbox)
                    if hand_lms:
                        hl = hand_lms[0]
                        hdet = True
                        hs = hand_d.check_sos_step(hs, hl)
                        debug_info = hand_d.get_debug_info(hl)
                        draw_hand_on_frame(frame, hl, (cx1, cy1),
                                         (cx2 - cx1, cy2 - cy1))
                except Exception:
                    pass

                if hdet:
                    hand_miss[tid] = 0
                else:
                    hand_miss[tid] = hand_miss.get(tid, 0) + 1
                    if hand_miss[tid] >= GRACE_FRAMES:
                        hs = max(0, hs - 1)
                        hand_miss[tid] = 0

                hand_states[tid] = hs
                draw_bbox_with_state(frame, bbox, tid, conf, hs)

                if debug_info:
                    panel_y = max(0, y1 - 130)
                    draw_debug_panel(frame, debug_info, hs,
                                   x_offset=max(0, x1), y_offset=panel_y)

                fingers = debug_info.get('fingers', {})
                row = [
                    frame_idx, round(t_sec, 3), tid, hs, int(hdet),
                    debug_info.get('palm_open', ''),
                    debug_info.get('thumb_in', ''),
                    debug_info.get('fingers_closed', ''),
                    debug_info.get('combined', ''),
                    debug_info.get('hand_span', ''),
                    debug_info.get('thumb_ratio', ''),
                    debug_info.get('thumb_idx_ratio', ''),
                ]
                for fname in ['index', 'middle', 'ring', 'pinky']:
                    fd = fingers.get(fname, {})
                    row.extend([
                        fd.get('ratio', ''),
                        fd.get('t1_closed', ''),
                        fd.get('t2_closed', ''),
                        fd.get('t3_closed', ''),
                    ])
                csv_writer.writerow(row)

                if hs == 3:
                    print(f"  SOS DETECTED! t={t_sec:.2f}s tid={tid}")
                    hand_states[tid] = 0

            draw_hud(frame, current_fps, frame_idx, t_sec, hand_states, len(assigned))
            cv2.putText(frame, f"t={t_sec:.2f}s", (w - 140, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if debug_writer:
                debug_writer.write(frame)

            if frame_idx % (int(src_fps) * 2) == 0:
                print(f"  frame={frame_idx} t={t_sec:.1f}s states={dict(hand_states)}")

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()
        if debug_writer:
            debug_writer.release()
        csv_file.close()

    elapsed = time.time() - t0
    print(f"\nDone -- {frame_idx} frames in {elapsed:.1f}s")
    print(f"CSV: {csv_path}")
    if save_video:
        print(f"Video: {out_dir}/")


def main():
    ap = argparse.ArgumentParser(
        description="Skeleton Debug Tool for RTSP/Video SOS Detection")
    ap.add_argument("--source", required=True,
                    help="Video file path or RTSP URL")
    ap.add_argument("--detector-url", default="http://127.0.0.1:8000")
    ap.add_argument("--out-dir", default="eval/debug_skeleton_out")
    ap.add_argument("--sample-fps", type=int, default=5)
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="Max seconds to process (0=unlimited)")
    ap.add_argument("--no-video", action="store_true",
                    help="Skip debug video output (CSV log only)")
    args = ap.parse_args()

    try:
        thresholds = yaml.safe_load(Path("config/thresholds.yaml").read_text())
    except Exception:
        thresholds = {}
    hand_cfg = thresholds.get("hand_sos", {})

    run_debug(
        source=args.source,
        detector_url=args.detector_url,
        hand_cfg=hand_cfg,
        out_dir=args.out_dir,
        sample_fps=args.sample_fps,
        save_video=not args.no_video,
        max_seconds=args.max_seconds,
    )


if __name__ == "__main__":
    main()
