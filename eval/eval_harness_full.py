#!/usr/bin/env python3
"""
eval/eval_harness_full.py — Full Detection Eval with Debug Visualization

ทดสอบ fall + hand_sos detection พร้อม debug video output ที่วาด:
  - 🟩 Bounding box (สีเปลี่ยนตาม state)
  - 🦴 Skeleton จาก keypoints 17 จุด
  - 🖐 Hand landmarks 21 จุด + state text
  - 📐 Spine angle + bbox ratio overlay
  - 📊 HUD panel (FPS, track count, hand/fall state)

Usage:
    python3 model_server.py &
    python3 eval/eval_harness_full.py \\
        --videos-dir eval/videos \\
        --labels-dir eval/labels \\
        --detector-url http://127.0.0.1:8000 \\
        --out-dir eval/report_debug \\
        --save-debug-video

    # ทดสอบเฉพาะไฟล์ RTSP:
    python3 eval/eval_harness_full.py --filter rtsp_hand
"""
import argparse
import csv
import json
import time
from collections import defaultdict, deque
from math import ceil
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import requests
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detectors.fall_detector import FallDetector
from detectors.hand_sos_detector import HandSOSDetector


# ═══════════════════════════════════════════════════════════════════════════════
# Skeleton drawing constants (COCO 17 keypoints)
# ═══════════════════════════════════════════════════════════════════════════════
SKELETON_CONNECTIONS = [
    (5, 6),    # L_shoulder - R_shoulder
    (5, 7),    # L_shoulder - L_elbow
    (7, 9),    # L_elbow - L_wrist
    (6, 8),    # R_shoulder - R_elbow
    (8, 10),   # R_elbow - R_wrist
    (5, 11),   # L_shoulder - L_hip
    (6, 12),   # R_shoulder - R_hip
    (11, 12),  # L_hip - R_hip
    (11, 13),  # L_hip - L_knee
    (13, 15),  # L_knee - L_ankle
    (12, 14),  # R_hip - R_knee
    (14, 16),  # R_knee - R_ankle
]

SKELETON_COLORS = {
    'torso': (0, 255, 200),     # cyan-green
    'left_arm': (255, 180, 0),  # orange
    'right_arm': (0, 180, 255), # light blue
    'left_leg': (200, 100, 255),# purple
    'right_leg': (100, 255, 100), # green
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


# ═══════════════════════════════════════════════════════════════════════════════
# Tracking + API
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}

    def _iou(self, a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        w = max(0, x2 - x1); h = max(0, y2 - y1)
        inter = w * h
        aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
        ab = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / (aa + ab - inter) if (aa + ab - inter) > 0 else 0.0

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
        tid_set = {d["track_id"] for d in dets}
        for tid in list(self.tracks):
            if tid not in tid_set:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > 5:
                    del self.tracks[tid]
        return dets


def detect_all(detector_url: str, frame: np.ndarray, timeout: int = 5) -> dict:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    try:
        r = requests.post(
            detector_url + "/detect_all",
            files={"image": ("f.jpg", buf.tobytes(), "image/jpeg")},
            timeout=timeout,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"[detect_all] error: {e}")
    return {"people": [], "objects": []}


def apply_clahe(frame):
    """CLAHE preprocessing — same as main.py L217-219"""
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
# Debug Drawing Functions
# ═══════════════════════════════════════════════════════════════════════════════

def draw_skeleton(frame, kps, conf_thresh=0.3):
    """Draw COCO-17 skeleton on frame."""
    if not kps or len(kps) < 17:
        return
    for a, b in SKELETON_CONNECTIONS:
        if a >= len(kps) or b >= len(kps):
            continue
        ka, kb = kps[a], kps[b]
        if len(ka) < 3 or len(kb) < 3:
            continue
        if ka[2] < conf_thresh or kb[2] < conf_thresh:
            continue
        pt1 = (int(ka[0]), int(ka[1]))
        pt2 = (int(kb[0]), int(kb[1]))
        color = _conn_color(a, b)
        cv2.line(frame, pt1, pt2, color, 2)

    # Draw keypoints
    for i, kp in enumerate(kps):
        if len(kp) < 3 or kp[2] < conf_thresh:
            continue
        cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 255, 255), -1)
        cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 0, 0), 1)


def draw_bbox(frame, bbox, tid, conf, fall_result, hand_state):
    """Draw bounding box with color based on state."""
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Color based on state
    if fall_result and fall_result.get("is_critical"):
        color = (0, 0, 255)  # RED - critical
        state_text = "CRITICAL"
    elif fall_result and fall_result.get("is_fallen"):
        color = (0, 0, 200)  # dark red - fallen
        state_text = "FALLEN"
    elif fall_result and fall_result.get("is_down"):
        color = (0, 140, 255)  # orange - down
        state_text = f"DOWN {fall_result.get('time_down', 0)}s"
    elif hand_state == 3:
        color = (0, 255, 0)  # GREEN - SOS confirmed
        state_text = "SOS!"
    elif hand_state >= 1:
        color = (255, 200, 0)  # cyan - hand in progress
        state_text = f"hand:{hand_state}/3"
    else:
        color = (0, 200, 0)  # green - normal
        state_text = "OK"

    # Draw bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw ID + conf
    label = f"ID:{tid} {conf:.0%} [{state_text}]"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Fall metrics below bbox
    if fall_result:
        metrics = (f"angle:{fall_result.get('spine_angle', 0):.0f}° "
                   f"ratio:{fall_result.get('bbox_ratio', 0):.1f} "
                   f"vel:{fall_result.get('vel_y_norm', 0):.3f}")
        cv2.putText(frame, metrics, (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return color


def draw_hand_landmarks(frame, hand_landmarks_list, w, h, hand_states_map, assigned):
    """Draw hand landmarks and match info."""
    if not hand_landmarks_list:
        return
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    for hi, hl in enumerate(hand_landmarks_list):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hl]
        # Draw connections
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (0, 220, 0), 1)
        # Draw points
        for pt in pts:
            cv2.circle(frame, pt, 3, (0, 255, 0), -1)
        # Hand center
        hcx = sum(p[0] for p in pts) / len(pts)
        hcy = sum(p[1] for p in pts) / len(pts)
        cv2.circle(frame, (int(hcx), int(hcy)), 6, (255, 0, 255), 2)
        cv2.putText(frame, f"hand#{hi}", (int(hcx) + 8, int(hcy) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)


def draw_hud(frame, fps, frame_idx, t_sec, hand_states, fall_states, predictions_count):
    """Draw HUD panel in top-left corner."""
    h, w = frame.shape[:2]
    panel_h = 120 + len(hand_states) * 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, min(panel_h, h)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 18
    cv2.putText(frame, f"FPS: {fps:.1f}  Frame: {frame_idx}  t={t_sec:.1f}s",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y += 20
    cv2.putText(frame, f"Detections: {predictions_count}",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
    y += 20

    # Per-track states
    for tid in sorted(set(list(hand_states.keys()) + list(fall_states.keys()))):
        hs = hand_states.get(tid, 0)
        fs = fall_states.get(tid, {})
        hand_labels = ["—", "palm✋", "thumb👍", "SOS🆘"]
        hl = hand_labels[hs] if hs < 4 else f"?{hs}"
        fall_text = ""
        if fs.get("is_fallen"):
            fall_text = " FALLEN"
        elif fs.get("is_down"):
            fall_text = f" DOWN({fs.get('time_down',0):.0f}s)"

        color = (0, 255, 0) if hs == 3 else (200, 200, 200)
        if fs.get("is_fallen"):
            color = (0, 0, 255)
        cv2.putText(frame, f"T{tid}: hand={hl}{fall_text}",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y += 18


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_video(video_path: Path, label: dict, detector_url: str,
                   fall_cfg: dict, hand_cfg: dict,
                   sample_fps: int = 5,
                   save_debug_video: bool = False,
                   debug_video_dir: Path = None) -> Tuple[List[dict], float]:
    """Process one video, run fall + hand_sos detection with debug viz."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️  cannot open {video_path}")
        return [], 0.0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = max(1, round(src_fps / sample_fps))

    fall_d = FallDetector(fall_cfg)
    hand_d = HandSOSDetector(hand_cfg)
    tracker = SimpleTracker()

    # Per-track state (same as main.py B2)
    hand_states: dict = {}
    hand_miss: dict = {}
    hand_first_seen: dict = {}
    hand_temporal_window = max(1, int(hand_cfg.get("temporal_window", 10)))
    hand_temporal_threshold = min(1.0, max(0.0, float(hand_cfg.get("temporal_threshold", 0.4))))
    hand_temporal_hits_required = max(1, ceil(hand_temporal_window * hand_temporal_threshold))
    hand_min_bbox_area_norm = max(0.0, float(hand_cfg.get("min_hand_bbox_area_norm", 0.0)))
    hand_min_track_age_seconds = max(0.0, float(hand_cfg.get("min_track_age_seconds", 0.8)))
    hand_recent_sos = defaultdict(lambda: deque(maxlen=hand_temporal_window))
    fall_states: dict = {}
    GRACE_FRAMES = 5

    cooldown_fall = {}
    cooldown_hand = {}
    hand_last_event_sec = -1e9
    COOLDOWN_SEC = fall_cfg.get("cooldown_seconds", 120)
    HAND_COOLDOWN = hand_cfg.get("cooldown_seconds", 60)

    # Debug video writer
    debug_writer = None
    if save_debug_video and debug_video_dir:
        debug_video_dir.mkdir(parents=True, exist_ok=True)
        out_path = debug_video_dir / f"debug_{video_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        debug_writer = cv2.VideoWriter(str(out_path), fourcc,
                                        min(sample_fps, src_fps),
                                        (frame_w, frame_h))
        print(f"  📹 Debug video → {out_path}")

    predictions = []
    frame_idx = 0
    t0 = time.time()
    fps_timer = []

    while True:
        ret, raw = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        t_sec = frame_idx / src_fps
        frame = raw.copy()
        h, w = frame.shape[:2]

        # FPS calculation
        now = time.time()
        fps_timer = [t for t in fps_timer if now - t < 1.0]
        fps_timer.append(now)
        current_fps = len(fps_timer)

        # ── Detect via model_server ──
        resp = detect_all(detector_url, frame)
        people_raw = resp.get("people", [])

        # ── Track ──
        assigned = tracker.update([{"bbox": p["bbox"]} for p in people_raw])
        alive = set(tracker.tracks.keys())
        dropped = set(hand_states.keys()) - alive
        for tid in dropped:
            for d in (hand_states, hand_miss, hand_first_seen, fall_states):
                d.pop(tid, None)
            hand_recent_sos.pop(tid, None)

        # ── CLAHE for hand detection (same as main.py) ──
        frame_clahe = apply_clahe(frame)

        # ── Per-person processing ──
        for i, det in enumerate(assigned):
            tid = det["track_id"]
            bbox = det["bbox"]
            kps = people_raw[i].get("keypoints", []) if i < len(people_raw) else []
            conf = people_raw[i].get("conf", 0.0) if i < len(people_raw) else 0.0

            # ── Fall Detection ──
            fr = None
            try:
                fr = fall_d.process(tid, kps, bbox, h, w)
                fall_states[tid] = fr
            except Exception as e:
                fr = {}

            is_confirmed = (fr.get("is_fallen") or fr.get("danger_lying")) and not fr.get("recovered_quickly")
            is_critical = fr.get("is_critical") and not fr.get("recovered_quickly")

            if (is_confirmed or is_critical) and tid not in cooldown_fall:
                cooldown_fall[tid] = t_sec
                predictions.append({
                    "video": video_path.name,
                    "event_type": "fall",
                    "track_id": tid,
                    "t_sec": round(t_sec, 2),
                    "confidence": round(max(0.5, conf), 2),
                    "matched": False,
                })
            elif not is_confirmed and tid in cooldown_fall:
                if t_sec - cooldown_fall[tid] > COOLDOWN_SEC:
                    del cooldown_fall[tid]

            # ── Hand SOS Detection (per-person crop strategy) ──
            # [RTSP FIX] ใช้ process_crop() + check_sos_step() ที่ตรงกับ main.py
            prev_hs = hand_states.get(tid, 0)
            hs = prev_hs
            hdet = False
            x1, y1, x2, y2 = [int(v) for v in bbox]
            if tid not in hand_first_seen:
                hand_first_seen[tid] = t_sec
            bbox_area_norm = (max(0.0, x2 - x1) * max(0.0, y2 - y1)) / max(1.0, float(w * h))
            track_age_sec = t_sec - hand_first_seen[tid]
            hand_eligible = (
                bbox_area_norm >= hand_min_bbox_area_norm and
                track_age_sec >= hand_min_track_age_seconds
            )

            # Expand bbox by 10% for hand detection margin
            bw, bh = x2 - x1, y2 - y1
            cx1 = max(0, x1 - int(bw * 0.1))
            cy1 = max(0, y1 - int(bh * 0.1))
            cx2 = min(w, x2 + int(bw * 0.1))
            cy2 = min(h, y2 + int(bh * 0.1))

            try:
                if hand_eligible:
                    hand_lms = hand_d.process_crop(frame_clahe, bbox)
                    if hand_lms:
                        hl = hand_lms[0]
                        hdet = True
                        hs = hand_d.check_sos_step(hs, hl)

                        # Draw hand landmarks on main frame (map crop coords back)
                        if save_debug_video and hand_d._results and hand_d._results.hand_landmarks:
                            crop_h = cy2 - cy1
                            crop_w = cx2 - cx1
                            for hi_idx, hl_draw in enumerate(hand_d._results.hand_landmarks):
                                pts_frame = [
                                    (cx1 + int(lm.x * crop_w),
                                     cy1 + int(lm.y * crop_h))
                                    for lm in hl_draw
                                ]
                                HAND_CONNS = [
                                    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                                    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                                    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
                                ]
                                for a, b in HAND_CONNS:
                                    if a < len(pts_frame) and b < len(pts_frame):
                                        cv2.line(frame, pts_frame[a], pts_frame[b], (0, 220, 0), 1)
                                for pt in pts_frame:
                                    cv2.circle(frame, pt, 3, (0, 255, 0), -1)
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
            is_sos_frame = bool(hdet and hs == 3 and hand_eligible)
            recent_sos = hand_recent_sos[tid]
            recent_sos.append(is_sos_frame)
            sos_hits = sum(1 for ok in recent_sos if ok)
            temporal_confirmed = len(recent_sos) >= hand_temporal_hits_required and sos_hits >= hand_temporal_hits_required
            fast_sos_confirmed = is_sos_frame and prev_hs >= 1
            sos_confirmed = temporal_confirmed or fast_sos_confirmed

            if sos_confirmed:
                if t_sec - hand_last_event_sec < HAND_COOLDOWN:
                    continue
                last_hand_event = cooldown_hand.get(tid, -1e9)
                if t_sec - last_hand_event >= HAND_COOLDOWN:
                    cooldown_hand[tid] = t_sec
                    hand_last_event_sec = t_sec
                    predictions.append({
                        "video": video_path.name,
                        "event_type": "hand_sos",
                        "track_id": tid,
                        "t_sec": round(t_sec, 2),
                        "confidence": 1.0,
                        "matched": False,
                    })

            # ── Debug Drawing ──
            if save_debug_video:
                draw_skeleton(frame, kps)
                draw_bbox(frame, bbox, tid, conf, fr, hs)

        # ── Draw HUD on debug frame ──
        if save_debug_video:
            draw_hud(frame, current_fps, frame_idx, t_sec,
                     hand_states, fall_states, len(predictions))

            # Timestamp
            cv2.putText(frame, f"t={t_sec:.2f}s", (w - 140, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if debug_writer:
                debug_writer.write(frame)

    cap.release()
    hand_d.release()
    if debug_writer:
        debug_writer.release()

    video_seconds = frame_idx / src_fps if src_fps else 0.0
    elapsed = time.time() - t0
    print(f"  {video_path.name}: {len(predictions)} predictions "
          f"in {elapsed:.1f}s (video={video_seconds:.1f}s)")
    return predictions, video_seconds


def match_predictions(predictions, gt_events, tolerance_sec=5.0):
    """Greedy matching — same logic as eval_augmented."""
    gt_used = [False] * len(gt_events)
    for pred in predictions:
        for i, gt in enumerate(gt_events):
            if gt_used[i]:
                continue
            if gt["event_type"] != pred["event_type"]:
                continue
            lo = gt["start_sec"] - tolerance_sec
            hi = gt["end_sec"] + tolerance_sec
            if lo <= pred["t_sec"] <= hi:
                pred["matched"] = True
                gt_used[i] = True
                break
    return predictions, gt_used


def compute_metrics(counts: dict) -> dict:
    metrics = {}
    for et, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)
              if recall not in (None, 0) and (precision + recall) > 0 else None)
        metrics[et] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Full Detection Eval with Debug Skeleton/BBox Visualization"
    )
    ap.add_argument("--videos-dir", default="eval/videos")
    ap.add_argument("--labels-dir", default="eval/labels")
    ap.add_argument("--detector-url", default="http://127.0.0.1:8000")
    ap.add_argument("--out-dir", default="eval/report_debug")
    ap.add_argument("--tolerance-sec", type=float, default=5.0)
    ap.add_argument("--sample-fps", type=int, default=5)
    ap.add_argument("--save-debug-video", action="store_true",
                    help="Save debug videos with skeleton/bbox overlay")
    ap.add_argument("--filter", default="",
                    help="Only process labels whose filename contains this string")
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_video_dir = out_dir / "debug_videos"

    # Load thresholds
    try:
        thresholds = yaml.safe_load(Path("config/thresholds.yaml").read_text())
    except Exception:
        thresholds = {}
    fall_cfg = thresholds.get("fall", {})
    hand_cfg = thresholds.get("hand_sos", {})

    # Collect label files (exclude object labels and multicam)
    label_files = sorted(
        p for p in labels_dir.glob("*.json")
        if p.name != "example.json"
        and not p.name.startswith("obj_")
        and not p.name.startswith("multicam_")
    )

    if args.filter:
        label_files = [p for p in label_files if args.filter in p.stem]
        print(f"🔍 Filter: '{args.filter}' → {len(label_files)} label files")

    if not label_files:
        print("⚠️  No label files found. Check --labels-dir and --filter")
        return

    all_predictions = []
    counts = {}
    total_video_seconds = 0.0

    print(f"\n{'='*70}")
    print(f"🎯 Full Detection Eval {'+ Debug Video' if args.save_debug_video else ''}")
    print(f"{'='*70}")
    print(f"Labels: {len(label_files)} files")
    print(f"Detector: {args.detector_url}")
    print(f"Sample FPS: {args.sample_fps}")
    print(f"{'='*70}\n")

    for label_path in label_files:
        label = json.loads(label_path.read_text(encoding="utf-8"))
        video_path = videos_dir / label["video"]
        if not video_path.exists():
            # Try resolving symlink
            resolved = video_path.resolve()
            if not resolved.exists():
                print(f"⚠️  missing video: {video_path}")
                continue

        print(f"▶ {label['video']} (location: {label.get('location', '?')}) ...")
        preds, vid_secs = evaluate_video(
            video_path, label, args.detector_url,
            fall_cfg, hand_cfg,
            sample_fps=args.sample_fps,
            save_debug_video=args.save_debug_video,
            debug_video_dir=debug_video_dir,
        )
        total_video_seconds += vid_secs

        gt_events = label.get("events", [])
        preds, gt_used = match_predictions(preds, gt_events, args.tolerance_sec)
        all_predictions.extend(preds)

        # Count per event type
        for ev in gt_events:
            et = ev["event_type"]
            counts.setdefault(et, {"tp": 0, "fp": 0, "fn": 0})
        for i, gt in enumerate(gt_events):
            et = gt["event_type"]
            if gt_used[i]:
                counts[et]["tp"] += 1
            else:
                counts[et]["fn"] += 1
        for pred in preds:
            et = pred["event_type"]
            counts.setdefault(et, {"tp": 0, "fp": 0, "fn": 0})
            if not pred["matched"]:
                counts[et]["fp"] += 1

    # ── Metrics ──
    metrics = compute_metrics(counts)

    total_fp = sum(c["fp"] for c in counts.values())
    hours = total_video_seconds / 3600.0
    false_alarms_per_hour = round(total_fp / hours, 2) if hours > 0 else None

    report = {
        "tolerance_sec": args.tolerance_sec,
        "total_video_hours": round(hours, 4),
        "false_alarms_per_hour": false_alarms_per_hour,
        "metrics": metrics,
        "config": {
            "fall": fall_cfg,
            "hand_sos": hand_cfg,
        },
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with open(out_dir / "raw_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video", "event_type", "track_id", "t_sec", "confidence", "matched",
        ])
        writer.writeheader()
        for p in all_predictions:
            writer.writerow(p)

    # ── Print summary ──
    print(f"\n{'='*70}")
    print(f"{'Event Type':<18}{'TP':>4}{'FP':>5}{'FN':>5}{'Precision':>12}{'Recall':>10}{'F1':>8}")
    print(f"{'-'*70}")
    for et, m in metrics.items():
        r = f"{m['recall']:.2f}" if m["recall"] is not None else "-"
        f1v = f"{m['f1']:.2f}" if m["f1"] is not None else "-"
        print(f"{et:<18}{m['tp']:>4}{m['fp']:>5}{m['fn']:>5}"
              f"{m['precision']:>12.2f}{r:>10}{f1v:>8}")
    print(f"{'-'*70}")
    print(f"Total video time: {hours:.4f} hr | False alarms/hour: {false_alarms_per_hour}")
    print(f"{'='*70}")
    print(f"\n📄 Report: {out_dir / 'report.json'}")
    print(f"📄 Predictions: {out_dir / 'raw_predictions.csv'}")
    if args.save_debug_video:
        print(f"🎬 Debug videos: {debug_video_dir}/")


if __name__ == "__main__":
    main()
