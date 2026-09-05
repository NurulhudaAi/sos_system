#!/usr/bin/env python3
"""
debug_fall_signals.py — Log FallGuard model output per-second to understand
why certain GT fall timestamps are not detected (FN analysis).

Usage:
    python eval/debug_fall_signals.py /path/to/video.mp4
"""
import sys
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detectors.fall_detector import FallDetector

# ── Reuse tracker from eval ──
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


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not video_path:
        print("Usage: python eval/debug_fall_signals.py /path/to/video.mp4")
        sys.exit(1)

    cfg_path = ROOT / "config" / "thresholds.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    general = cfg.get("general", {})

    from ultralytics import YOLO
    yolo_model = general.get("yolo_model", "yolov8n-pose.pt")
    fall_pose_model = general.get("fall_pose_model")
    person_conf = general.get("person_conf", 0.30)
    fall_pose_conf = general.get("fall_pose_conf", person_conf)
    class_map = {int(k): v for k, v in general.get("fall_pose_class_map", {0: "laying", 1: "standing"}).items()}
    frame_skip = general.get("frame_skip", 1)

    print(f"[debug] Loading generic model: {yolo_model}")
    yolo = YOLO(yolo_model)
    yolo_fall_pose = None
    if fall_pose_model:
        print(f"[debug] Loading FallGuard model: {fall_pose_model}")
        yolo_fall_pose = YOLO(fall_pose_model)

    fall_d = FallDetector(cfg.get("fall", {}))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration = total / fps if fps else 0
    print(f"[debug] Video: {w}x{h}, {fps:.1f}fps, {duration:.1f}s")

    tracker = SimpleTracker(frame_w=w, frame_h=h)

    # GT timestamps of interest (for highlighting)
    gt_fall_ts = [24, 45, 53, 88, 142, 151]
    gt_hand_ts = [91, 156]

    frame_idx = 0
    last_printed_sec = -1

    print()
    print(f"{'sec':>6s}  {'tid':>3s}  {'posture':>10s}  {'pconf':>5s}  "
          f"{'ratio':>5s}  {'angle':>5s}  {'is_down':>7s}  {'ground_t':>8s}  "
          f"{'time_dn':>7s}  {'geom_t':>6s}  {'model_t':>7s}  "
          f"{'is_fallen':>9s}  {'recov':>5s}  {'note':s}")
    print("-" * 130)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % max(1, frame_skip) != 0:
            continue
        ts_sec = frame_idx / fps

        # Run FallGuard model
        people = []
        if yolo_fall_pose is not None:
            results = yolo_fall_pose(frame, conf=fall_pose_conf, verbose=False)
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                kpts = getattr(results[0], "keypoints", None)
                xy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy().tolist()
                classes = boxes.cls.cpu().numpy().astype(int).tolist()
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
                for i, box in enumerate(xy.tolist()):
                    x1, y1, x2, y2 = [float(v) for v in box]
                    conf = float(confs[i]) if i < len(confs) else 0.0
                    kp = kpts_all[i] if i < len(kpts_all) else []
                    pc = class_map.get(classes[i], str(classes[i])) if i < len(classes) else None
                    people.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf,
                        "keypoints": kp,
                        "posture_class": pc,
                        "posture_conf": conf,
                    })

        assigned = tracker.update(people) if people else []

        for p in assigned:
            tid = p["track_id"]
            bbox = p["bbox"]
            kp = np.array(p.get("keypoints") or np.zeros((17, 3)))
            posture_class = p.get("posture_class")
            posture_conf = float(p.get("posture_conf", 0.0))

            fr = fall_d.process(tid, kp, bbox, h, w,
                                posture_class=posture_class,
                                posture_conf=posture_conf)

            current_sec = int(ts_sec)
            # Print once per second, or always print when near GT timestamps or when fallen
            near_gt = any(abs(ts_sec - gt) <= 6 for gt in gt_fall_ts + gt_hand_ts)
            state_change = fr["is_fallen"] or fr["is_down"]

            if current_sec != last_printed_sec or near_gt or state_change:
                note = ""
                for gt in gt_fall_ts:
                    if abs(ts_sec - gt) <= 1:
                        note = "◄ GT:FALL"
                        break
                for gt in gt_hand_ts:
                    if abs(ts_sec - gt) <= 1:
                        note = "◄ GT:HAND_SOS"
                        break
                if fr["is_fallen"]:
                    note += " ★FALLEN★"

                print(f"{ts_sec:6.1f}  {tid:3d}  {str(posture_class):>10s}  "
                      f"{posture_conf:5.2f}  "
                      f"{fr['bbox_ratio']:5.2f}  {fr['spine_angle']:5.1f}  "
                      f"{str(fr['is_down']):>7s}  "
                      f"{'yes' if fr['ground_time'] else 'no':>8s}  "
                      f"{fr['time_down']:7.1f}  "
                      f"{fr['geometry_time']:6.1f}  "
                      f"{fr['model_lying_time']:7.1f}  "
                      f"{str(fr['is_fallen']):>9s}  "
                      f"{str(fr['recovered_quickly']):>5s}  "
                      f"{note}")
                if current_sec != last_printed_sec:
                    last_printed_sec = current_sec

    cap.release()
    print("\n[debug] Done.")


if __name__ == "__main__":
    main()
