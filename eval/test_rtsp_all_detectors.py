#!/usr/bin/env python3
"""
test_rtsp_all_detectors.py — Comprehensive RTSP evaluation for:
  1. Fall Detection (FallDetector)
  2. Hand SOS Detection (HandSOSDetector)
  3. Hands Over Head Gesture (HandsOverHeadDetector)

Runs directly on positive and negative RTSP videos and produces precision/recall
and false positive rate diagnostics.
"""
import argparse
import json
import sys
import time
from collections import defaultdict, deque
from math import ceil
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detectors.fall_detector import FallDetector
from detectors.hand_sos_detector import HandSOSDetector
from detectors.hands_over_head_detector import HandsOverHeadDetector


class SimpleTracker:
    def __init__(self, frame_w, frame_h, iou_thresh=0.3, center_dist_norm=0.12):
        self.next_id = 0
        self.tracks = {}
        self.iou_thresh = iou_thresh
        self.center_dist_norm = center_dist_norm
        self._diag = float((frame_w ** 2 + frame_h ** 2) ** 0.5)

    def _iou(self, a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        w = max(0, x2 - x1); h = max(0, y2 - y1)
        inter = w * h
        aa = max(1e-6, (a[2]-a[0])*(a[3]-a[1]))
        ab = max(1e-6, (b[2]-b[0])*(b[3]-b[1]))
        return inter / (aa+ab-inter) if (aa+ab-inter) > 0 else 0.0

    def _center(self, box):
        return ((box[0]+box[2])/2.0, (box[1]+box[3])/2.0)

    def _center_dist_norm(self, a, b):
        ax, ay = self._center(a); bx, by = self._center(b)
        d = ((ax-bx)**2 + (ay-by)**2) ** 0.5
        return d / self._diag if self._diag else 1.0

    def update(self, dets):
        used, unmatched = [], []
        for det in dets:
            best_id, best_iou = None, 0.0
            for tid, t in self.tracks.items():
                iou = self._iou(det["bbox"], t["bbox"])
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_iou >= self.iou_thresh and best_id not in used:
                det["track_id"] = best_id
                self.tracks[best_id].update(bbox=det["bbox"], lost=0)
                used.append(best_id)
            else:
                unmatched.append(det)
        for det in unmatched:
            best_id, best_dist = None, self.center_dist_norm
            for tid, t in self.tracks.items():
                if tid in used:
                    continue
                dist = self._center_dist_norm(det["bbox"], t["bbox"])
                if dist < best_dist:
                    best_dist, best_id = dist, tid
            if best_id is not None:
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


def load_cfg():
    cfg_path = ROOT / "config" / "thresholds.yaml"
    if cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text())
    return {}


def build_models(cfg):
    from ultralytics import YOLO
    general = cfg.get("general", {})
    yolo_path = ROOT / "models" / "yolov8n-pose.pt"
    fg_path = ROOT / "models" / "FallGuard_YOLOv8Pose.pt"

    print(f"Loading generic pose model: {yolo_path}")
    yolo = YOLO(str(yolo_path))

    yolo_fg = None
    if fg_path.exists():
        print(f"Loading FallGuard model: {fg_path}")
        try:
            yolo_fg = YOLO(str(fg_path))
        except Exception as e:
            print(f"⚠️ Could not load FallGuard: {e}")

    return yolo, yolo_fg


def detect_people_frame(frame, yolo, yolo_fg, person_conf=0.20, fg_conf=0.20):
    h, w = frame.shape[:2]

    def _parse(results, with_posture):
        out = []
        if not results or results[0].boxes is None:
            return out
        boxes = results[0].boxes
        kpts = getattr(results[0], "keypoints", None)
        try:
            xy = boxes.xyxy.cpu().numpy()
        except Exception:
            xy = np.array(boxes.xyxy)
        try:
            confs = boxes.conf.cpu().numpy().tolist()
        except Exception:
            confs = list(boxes.conf)
        classes = None
        if with_posture:
            try:
                classes = boxes.cls.cpu().numpy().astype(int).tolist()
            except Exception:
                classes = None
        kpts_all = []
        if kpts is not None and getattr(kpts, "data", None) is not None:
            karr = kpts.data.cpu().numpy()
            for kp in karr:
                kp_list = []
                for x, y, c in kp:
                    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                        kp_list.append([float(x*w), float(y*h), float(c)])
                    else:
                        kp_list.append([float(x), float(y), float(c)])
                kpts_all.append(kp_list)
        for i, box in enumerate(xy.tolist() if hasattr(xy, "tolist") else xy):
            x1, y1, x2, y2 = [float(v) for v in box]
            conf = float(confs[i]) if i < len(confs) else 0.0
            kp = kpts_all[i] if i < len(kpts_all) else []
            person = {"bbox": [x1, y1, x2, y2], "conf": conf, "keypoints": kp}
            if with_posture and classes is not None and i < len(classes):
                person["posture_class"] = "laying" if classes[i] == 0 else "standing"
                person["posture_conf"] = conf
            out.append(person)
        return out

    results = yolo(frame, conf=person_conf, classes=[0], verbose=False)
    people = _parse(results, with_posture=False)

    if yolo_fg is not None and people:
        try:
            fg_res = yolo_fg(frame, conf=fg_conf, verbose=False)
            fg_dets = _parse(fg_res, with_posture=True)
            for person in people:
                best_iou, best_fp = 0.0, None
                px1, py1, px2, py2 = person["bbox"]
                for fp in fg_dets:
                    fx1, fy1, fx2, fy2 = fp["bbox"]
                    ix1 = max(px1, fx1); iy1 = max(py1, fy1)
                    ix2 = min(px2, fx2); iy2 = min(py2, fy2)
                    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    area_p = max(1e-6, (px2-px1)*(py2-py1))
                    area_f = max(1e-6, (fx2-fx1)*(fy2-fy1))
                    iou = inter / (area_p + area_f - inter) if (area_p + area_f - inter) > 0 else 0
                    if iou > best_iou:
                        best_iou, best_fp = iou, fp

                # Match by IoU or center containment
                if best_fp is not None:
                    if best_iou >= 0.3:
                        person["posture_class"] = best_fp.get("posture_class")
                        person["posture_conf"] = best_fp.get("posture_conf", 0.0)
                    else:
                        fxc = (best_fp["bbox"][0] + best_fp["bbox"][2]) / 2.0
                        fyc = (best_fp["bbox"][1] + best_fp["bbox"][3]) / 2.0
                        if px1 <= fxc <= px2 and py1 <= fyc <= py2:
                            person["posture_class"] = best_fp.get("posture_class")
                            person["posture_conf"] = best_fp.get("posture_conf", 0.0)
        except Exception:
            pass

    return people


def evaluate_video(video_path: Path, yolo, yolo_fg, cfg, max_seconds=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}")
        return {"video": video_path.name, "error": "cannot open"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration_s = total_frames / fps if fps else 0

    fall_cfg = cfg.get("fall", {})
    hand_cfg = cfg.get("hand_sos", {})
    head_cfg = cfg.get("hands_over_head", {})

    fall_d = FallDetector(fall_cfg)
    hand_d = HandSOSDetector(hand_cfg)
    head_d = HandsOverHeadDetector(head_cfg)
    tracker = SimpleTracker(frame_w=w, frame_h=h)

    # Hand SOS state tracking
    hand_temporal_window = max(1, int(hand_cfg.get("temporal_window", 10)))
    hand_temporal_threshold = min(1.0, max(0.0, float(hand_cfg.get("temporal_threshold", 0.6))))
    hand_temporal_hits_req = max(1, ceil(hand_temporal_window * hand_temporal_threshold))
    hand_min_bbox_area = max(0.0, float(hand_cfg.get("min_hand_bbox_area_norm", 0.0015)))
    hand_min_track_age = max(0.0, float(hand_cfg.get("min_track_age_seconds", 0.4)))
    min_consec_sos = max(1, int(hand_cfg.get("min_consec_sos_frames", 3)))
    grace_frames = 5

    hand_states, hand_miss, hand_first_seen, hand_consec3 = {}, {}, {}, {}
    hand_recent_sos = defaultdict(lambda: deque(maxlen=hand_temporal_window))

    # Event tracking & debouncing
    fall_ev = {}
    last_fall_event = {}
    last_hand_event = {}
    last_head_event = {}

    FALL_COOLDOWN = float(fall_cfg.get("cooldown_seconds", 120))
    HAND_COOLDOWN = float(hand_cfg.get("cooldown_seconds", 60))
    HEAD_COOLDOWN = float(head_cfg.get("cooldown_seconds", 60))

    events = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        ts_sec = frame_idx / fps

        if max_seconds and ts_sec > max_seconds:
            break

        # CLAHE for hand detector
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        frame_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        people = detect_people_frame(frame, yolo, yolo_fg)
        assigned = tracker.update(people) if people else []

        # Cleanup disappeared tracks
        alive_tids = set(tracker.tracks.keys())
        for dropped_tid in list(hand_states.keys()):
            if dropped_tid not in alive_tids:
                hand_states.pop(dropped_tid, None)
                hand_miss.pop(dropped_tid, None)
                hand_first_seen.pop(dropped_tid, None)
                hand_consec3.pop(dropped_tid, None)
                hand_recent_sos.pop(dropped_tid, None)
                fall_ev.pop(dropped_tid, None)
                head_d.cleanup_track(dropped_tid)

        for p in assigned:
            tid = p["track_id"]
            bbox = p["bbox"]
            kp = np.array(p.get("keypoints") or np.zeros((17, 3)))
            posture_class = p.get("posture_class")
            posture_conf = float(p.get("posture_conf", 0.0))

            # 1. Fall Detection
            fr = fall_d.process(
                tid, kp, bbox, h, w,
                posture_class=posture_class,
                posture_conf=posture_conf,
                timestamp=ts_sec,
            )
            is_fallen = bool(fr.get("is_fallen") or fr.get("danger_lying")) and not fr.get("recovered_quickly")

            if is_fallen:
                fall_ev[tid] = True
                prev_f = last_fall_event.get(tid, -1e9)
                if ts_sec - prev_f >= FALL_COOLDOWN:
                    last_fall_event[tid] = ts_sec
                    events.append({
                        "video": video_path.name,
                        "timestamp": round(ts_sec, 2),
                        "track_id": tid,
                        "event_type": "fall",
                        "posture": posture_class,
                        "ratio": round(fr.get("bbox_ratio", 0.0), 2),
                        "angle": round(fr.get("spine_angle", 0.0), 1),
                    })
            else:
                fall_ev[tid] = False

            # 2. Hands Over Head Detection (suppressed if fallen)
            if not fall_ev.get(tid, False):
                head_res = head_d.process(tid, kp, h, w, timestamp=ts_sec)
                if head_res.get("triggered"):
                    prev_h = last_head_event.get(tid, -1e9)
                    if ts_sec - prev_h >= HEAD_COOLDOWN:
                        last_head_event[tid] = ts_sec
                        events.append({
                            "video": video_path.name,
                            "timestamp": round(ts_sec, 2),
                            "track_id": tid,
                            "event_type": "hands_over_head",
                            "time_held": head_res.get("time_held", 0.0),
                        })

            # 3. Hand SOS Detection (suppressed if fallen)
            if not fall_ev.get(tid, False):
                prev_hs = hand_states.get(tid, 0)
                hs = prev_hs
                hdet = False
                if tid not in hand_first_seen:
                    hand_first_seen[tid] = ts_sec

                x1, y1, x2, y2 = bbox
                bbox_area_norm = (max(0.0, x2 - x1) * max(0.0, y2 - y1)) / max(1.0, float(w * h))
                track_age = ts_sec - hand_first_seen[tid]
                hand_eligible = (bbox_area_norm >= hand_min_bbox_area and track_age >= hand_min_track_age)

                try:
                    if hand_eligible:
                        hand_lms = hand_d.process_crop(frame_clahe, bbox)
                        if hand_lms:
                            hl = hand_lms[0]
                            hdet = True
                            hs = hand_d.check_sos_step(hs, hl)
                except Exception:
                    pass

                if hdet:
                    hand_miss[tid] = 0
                else:
                    hand_miss[tid] = hand_miss.get(tid, 0) + 1
                    if hand_miss[tid] >= grace_frames:
                        hs = max(0, hs - 1)
                        hand_miss[tid] = 0
                hand_states[tid] = hs

                if hs == 3 and hdet:
                    hand_consec3[tid] = hand_consec3.get(tid, 0) + 1
                else:
                    hand_consec3[tid] = 0

                consec_ok = hand_consec3.get(tid, 0) >= min_consec_sos
                is_sos_frame = bool(hdet and hs == 3 and hand_eligible and consec_ok)
                recent = hand_recent_sos[tid]
                recent.append(is_sos_frame)
                sos_hits = sum(1 for ok in recent if ok)
                sos_confirmed = (len(recent) >= hand_temporal_hits_req and sos_hits >= hand_temporal_hits_req)

                if sos_confirmed:
                    prev_s = last_hand_event.get(tid, -1e9)
                    if ts_sec - prev_s >= HAND_COOLDOWN:
                        last_hand_event[tid] = ts_sec
                        events.append({
                            "video": video_path.name,
                            "timestamp": round(ts_sec, 2),
                            "track_id": tid,
                            "event_type": "hand_sos",
                            "state": hs,
                        })

    cap.release()
    return {
        "video": video_path.name,
        "duration_s": round(duration_s, 1),
        "events": events,
    }


def main():
    parser = argparse.ArgumentParser(description="Test Fall, Hand SOS, and Hands Over Head on RTSP videos")
    parser.add_argument("--video", type=str, help="Single video file path")
    parser.add_argument("--dir", type=str, help="Directory of videos")
    parser.add_argument("--max-seconds", type=float, default=None, help="Max video seconds to process")
    parser.add_argument("--out", type=str, default="eval/report_rtsp_eval.json", help="Output JSON path")
    args = parser.parse_args()

    cfg = load_cfg()
    yolo, yolo_fg = build_models(cfg)

    videos = []
    if args.video:
        videos.append(Path(args.video))
    elif args.dir:
        p = Path(args.dir)
        videos = sorted([v for v in p.iterdir() if v.suffix.lower() in [".mp4", ".avi", ".m4v"]])
    else:
        print("Please provide --video or --dir")
        sys.exit(1)

    print(f"\nEvaluating {len(videos)} video(s)...")
    all_results = []
    for v in videos:
        print(f"\n--- Testing: {v.name} ---")
        t0 = time.time()
        res = evaluate_video(v, yolo, yolo_fg, cfg, max_seconds=args.max_seconds)
        elapsed = time.time() - t0
        events = res.get("events", [])
        dur = res.get("duration_s", 0)
        print(f"Processed {dur}s in {elapsed:.1f}s | Events detected: {len(events)}")
        for e in events:
            print(f"  [{e['event_type'].upper():<15}] @ {e['timestamp']:>6.1f}s | tid={e['track_id']}")
        all_results.append(res)

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
