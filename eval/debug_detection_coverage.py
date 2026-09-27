#!/usr/bin/env python3
"""
debug_detection_coverage.py — Check how many people FallGuard model detects
per second across the entire video. Shows gaps where nobody is detected.

Usage:
    python eval/debug_detection_coverage.py /path/to/video.mp4
"""
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not video_path:
        print("Usage: python eval/debug_detection_coverage.py /path/to/video.mp4")
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
    print(f"[debug] person_conf={person_conf}, fall_pose_conf={fall_pose_conf}")
    print()

    # ── Compare FallGuard vs Generic model per second ──
    print(f"{'sec':>6s}  {'FG_cnt':>6s}  {'FG_cls':>20s}  {'FG_conf':>20s}  "
          f"{'GEN_cnt':>7s}  {'GEN_conf':>20s}  {'note':s}")
    print("-" * 120)

    gt_ts = [24, 45, 53, 88, 91, 142, 151, 156]
    frame_idx = 0
    last_sec = -1

    # Stats
    fg_no_detect_ranges = []
    fg_gap_start = None
    total_secs_fg_none = 0
    total_secs_gen_none = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % max(1, frame_skip) != 0:
            continue
        ts_sec = frame_idx / fps
        current_sec = int(ts_sec)

        # Only print once per second
        if current_sec == last_sec:
            continue
        last_sec = current_sec

        # ── FallGuard model ──
        fg_count = 0
        fg_classes = []
        fg_confs = []
        if yolo_fall_pose is not None:
            results = yolo_fall_pose(frame, conf=fall_pose_conf, verbose=False)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                fg_count = len(boxes)
                try:
                    cls_list = boxes.cls.cpu().numpy().astype(int).tolist()
                    conf_list = boxes.conf.cpu().numpy().tolist()
                    fg_classes = [class_map.get(c, str(c)) for c in cls_list]
                    fg_confs = [round(c, 2) for c in conf_list]
                except:
                    pass

        # ── Generic YOLO model (person class=0) ──
        gen_count = 0
        gen_confs = []
        results_gen = yolo(frame, conf=person_conf, classes=[0], verbose=False)
        if results_gen and results_gen[0].boxes is not None and len(results_gen[0].boxes) > 0:
            gen_count = len(results_gen[0].boxes)
            try:
                gen_confs = [round(c, 2) for c in results_gen[0].boxes.conf.cpu().numpy().tolist()]
            except:
                pass

        # Track FallGuard gaps
        if fg_count == 0:
            if fg_gap_start is None:
                fg_gap_start = current_sec
            total_secs_fg_none += 1
        else:
            if fg_gap_start is not None:
                fg_no_detect_ranges.append((fg_gap_start, current_sec - 1))
                fg_gap_start = None

        if gen_count == 0:
            total_secs_gen_none += 1

        # Note
        note = ""
        for gt in gt_ts:
            if abs(current_sec - gt) <= 1:
                note = "◄ GT"
                break
        if fg_count == 0 and gen_count > 0:
            note += " ⚠️FG_MISS"
        if fg_count < gen_count:
            note += f" ⚠️FG<GEN"

        # Print all seconds (compact, one per second)
        fg_cls_str = ",".join(fg_classes[:3]) if fg_classes else "-"
        fg_conf_str = ",".join(str(c) for c in fg_confs[:3]) if fg_confs else "-"
        gen_conf_str = ",".join(str(c) for c in gen_confs[:3]) if gen_confs else "-"

        print(f"{current_sec:6d}  {fg_count:6d}  {fg_cls_str:>20s}  {fg_conf_str:>20s}  "
              f"{gen_count:7d}  {gen_conf_str:>20s}  {note}")

    if fg_gap_start is not None:
        fg_no_detect_ranges.append((fg_gap_start, current_sec))

    cap.release()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_secs = int(duration)
    print(f"Total video seconds: {total_secs}")
    print(f"FallGuard: no detection in {total_secs_fg_none}/{total_secs}s ({100*total_secs_fg_none/max(1,total_secs):.0f}%)")
    print(f"Generic:   no detection in {total_secs_gen_none}/{total_secs}s ({100*total_secs_gen_none/max(1,total_secs):.0f}%)")
    print()
    if fg_no_detect_ranges:
        print("FallGuard detection gaps (no person detected):")
        for s, e in fg_no_detect_ranges:
            dur = e - s + 1
            gt_in_range = [t for t in gt_ts if s <= t <= e]
            gt_str = f"  ← CONTAINS GT: {gt_in_range}" if gt_in_range else ""
            print(f"  {s:>4d}s – {e:>4d}s  ({dur:>3d}s gap){gt_str}")
    print()
    print("[debug] Done.")


if __name__ == "__main__":
    main()
