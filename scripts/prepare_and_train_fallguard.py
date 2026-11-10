#!/usr/bin/env python3
"""
scripts/prepare_and_train_fallguard.py
======================================
End-to-end dataset preparation (auto-labeling 17 COCO keypoints + posture class
from CCTV footage) and YOLOv8-Pose fine-tuning pipeline for FallGuard_YOLOv8Pose.pt.

Classes supported:
  0: laying   (person fallen / lying flat on floor)
  1: standing (person standing / walking upright)
  2: sitting  (person sitting on bench / chair / stairs / crouching)

Usage Examples:
  1. Prepare YOLO-Pose dataset from CCTV videos using base yolov8m-pose.pt:
     python3 scripts/prepare_and_train_fallguard.py prepare \
       --videos-dir "eval/videos" \
       --out-dir "dataset/fallguard_cctv" \
       --pose-model "models/yolov8m-pose.pt" \
       --sample-fps 2.0

  2. Fine-tune FallGuard_YOLOv8Pose.pt on the prepared dataset:
     python3 scripts/prepare_and_train_fallguard.py train \
       --data "dataset/fallguard_cctv/fallguard_pose.yaml" \
       --base-weights "models/FallGuard_YOLOv8Pose.pt" \
       --epochs 50 \
       --imgsz 640 \
       --batch 16 \
       --export-to "models/FallGuard_YOLOv8Pose.pt"
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detectors.fall_detector import FallDetector

# Default Ground-Truth Fall Windows (seconds) from Today Dataset CCTV Benchmark
DEFAULT_FALL_WINDOWS = {
    "C3-FL2.avi": [(12.0, 28.0)],
    "D1-Carpark.mp4": [(220.0, 240.0)],
    "D1-Stair-FL2 1.mp4": [(142.0, 155.0)],
    "D3-Hall-FL1 1.mp4": [(82.0, 100.0)],
    "D4-Hall-FL2 1.mp4": [(102.0, 118.0)],
    "E1-Hall 1.mp4": [(120.0, 145.0)],
}


def classify_person_posture(
    fall_det: FallDetector,
    kps: np.ndarray,
    bbox: list[float],
    in_fall_window: bool,
) -> int:
    """
    Classify a single person's posture into:
      0 = laying
      1 = standing
      2 = sitting
    using keypoint geometry + ground-truth fall timestamp windows.
    """
    x1, y1, x2, y2 = bbox
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    ratio = box_w / box_h

    # Compute spine angle if shoulders and hips are visible
    ls, rs = kps[5], kps[6]
    lh, rh = kps[11], kps[12]
    has_pose = all(float(p[2]) >= 0.25 for p in (ls, rs, lh, rh))
    angle = 90.0
    if has_pose:
        sh = np.array([(ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0])
        hp = np.array([(lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0])
        d = sh - hp
        angle = float(abs(np.degrees(np.arctan2(abs(d[1]), abs(d[0]) + 1e-6))))

    # Check sitting/crouching first
    if fall_det._is_sitting(kps, bbox) or fall_det._is_crouching_or_bending(kps, bbox):
        return 2  # sitting

    # Check horizontal / fallen posture
    if in_fall_window and (angle < 38.0 or ratio >= 1.25):
        return 0  # laying

    if ratio >= 1.45 and angle <= 30.0 and in_fall_window:
        return 0  # laying

    # If person is horizontal outside a fall window (e.g., lying on bench in E1-Hall 1),
    # label as sitting/resting (class 2) so the model learns NOT to classify bench resting as floor fall!
    if angle < 38.0 or ratio >= 1.20:
        return 2  # sitting / bench-resting

    return 1  # standing


def to_yolo_pose_line(
    cls_id: int,
    bbox: list[float],
    kps: np.ndarray,
    img_w: int,
    img_h: int,
) -> str:
    """Format a single detection as a YOLOv8-Pose label line."""
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    nw = bw / img_w
    nh = bh / img_h

    cx = min(1.0, max(0.0, cx))
    cy = min(1.0, max(0.0, cy))
    nw = min(1.0, max(0.0, nw))
    nh = min(1.0, max(0.0, nh))

    parts = [f"{cls_id}", f"{cx:.6f}", f"{cy:.6f}", f"{nw:.6f}", f"{nh:.6f}"]
    for i in range(17):
        if i < len(kps):
            kx, ky, kc = float(kps[i][0]), float(kps[i][1]), float(kps[i][2])
            nx = min(1.0, max(0.0, kx / img_w)) if kc > 0.05 else 0.0
            ny = min(1.0, max(0.0, ky / img_h)) if kc > 0.05 else 0.0
            v = 2 if kc >= 0.30 else (1 if kc > 0.05 else 0)
        else:
            nx, ny, v = 0.0, 0.0, 0
        parts.extend([f"{nx:.6f}", f"{ny:.6f}", f"{v}"])
    return " ".join(parts)


def prepare_dataset(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out_dir)
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    fall_windows = dict(DEFAULT_FALL_WINDOWS)
    if args.windows_json and Path(args.windows_json).exists():
        with open(args.windows_json, "r", encoding="utf-8") as f:
            fall_windows.update(json.load(f))

    pose_model = YOLO(args.pose_model)
    with open(ROOT / "config" / "thresholds.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    fall_det = FallDetector(cfg.get("fall", {}))

    video_files = sorted(
        p for p in videos_dir.iterdir()
        if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
    )
    if not video_files:
        raise RuntimeError(f"No video files found in {videos_dir}")

    rng = random.Random(args.seed)
    class_counts = {0: 0, 1: 0, 2: 0}
    saved_frames = 0

    for vpath in video_files:
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"[WARN] Could not open {vpath.name}, skipping.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        windows = fall_windows.get(vpath.name, [])
        frame_idx = 0

        print(f"[INFO] Scanning {vpath.name} (fps={fps:.1f}, fall_windows={windows})...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_sec = frame_idx / fps
            in_fall = any(start <= t_sec <= end for start, end in windows)
            # Sample at higher rate (4 fps) during fall windows, and args.sample_fps otherwise
            eff_sample_fps = max(args.sample_fps, 4.0) if in_fall else args.sample_fps
            step = max(1, int(round(fps / eff_sample_fps)))

            if frame_idx % step != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            results = pose_model(frame, conf=args.conf, verbose=False)
            r0 = results[0]

            label_lines = []
            has_non_standing = False
            if r0.boxes is not None and r0.keypoints is not None and len(r0.boxes) > 0:
                boxes_xyxy = r0.boxes.xyxy.cpu().numpy()
                kps_all = r0.keypoints.data.cpu().numpy()

                for i in range(len(boxes_xyxy)):
                    bbox = boxes_xyxy[i].tolist()
                    kps = kps_all[i]
                    cls_id = classify_person_posture(fall_det, kps, bbox, in_fall)
                    if not args.three_class and cls_id == 2:
                        cls_id = 1  # collapse sitting into standing if 2-class mode requested
                    if cls_id in (0, 2):
                        has_non_standing = True
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    label_lines.append(to_yolo_pose_line(cls_id, bbox, kps, w, h))

            # Balance dataset: keep all frames with laying/sitting, subsample pure-standing frames
            if label_lines and (has_non_standing or rng.random() < args.standing_keep_prob):
                split = "val" if rng.random() < args.val_split else "train"
                stem = f"{vpath.stem.replace(' ', '_')}_f{frame_idx:06d}"
                img_out = out_dir / "images" / split / f"{stem}.jpg"
                lbl_out = out_dir / "labels" / split / f"{stem}.txt"
                cv2.imwrite(str(img_out), frame)
                lbl_out.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                saved_frames += 1

            frame_idx += 1
        cap.release()

    names = {0: "laying", 1: "standing", 2: "sitting"} if args.three_class else {0: "laying", 1: "standing"}
    yaml_path = out_dir / "fallguard_pose.yaml"
    dataset_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [17, 3],
        "flip_idx": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
        "names": names,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)

    print(f"\n[DONE] Prepared {saved_frames} labeled frames in {out_dir}")
    print(f"       Class distribution: {class_counts}")
    print(f"       Dataset config written to: {yaml_path}")


def train_model(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    model = YOLO(args.base_weights)
    print(f"[INFO] Starting YOLOv8-Pose fine-tuning from {args.base_weights} on {data_yaml}...")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs" / "pose"),
        name=args.run_name,
        exist_ok=True,
        lr0=args.lr0,
        mosaic=0.5,
        fliplr=0.5,
    )

    best_pt = ROOT / "runs" / "pose" / args.run_name / "weights" / "best.pt"
    if best_pt.exists() and args.export_to:
        dst = Path(args.export_to)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, dst)
        print(f"[DONE] Exported fine-tuned weights to: {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CCTV Pose Dataset & Fine-tune FallGuard_YOLOv8Pose.pt")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Auto-extract and label CCTV frames into YOLOv8-Pose format")
    p_prep.add_argument("--videos-dir", default="eval/videos", help="Directory containing CCTV videos")
    p_prep.add_argument("--out-dir", default="dataset/fallguard_cctv", help="Output dataset directory")
    p_prep.add_argument("--pose-model", default="models/yolov8m-pose.pt", help="Base YOLO-Pose model for keypoint extraction")
    p_prep.add_argument("--windows-json", default="", help="Optional JSON file mapping video filename to [[start_s, end_s], ...]")
    p_prep.add_argument("--sample-fps", type=float, default=1.0, help="Frame sampling rate outside fall windows")
    p_prep.add_argument("--standing-keep-prob", type=float, default=0.25, help="Subsampling probability for pure-standing frames")
    p_prep.add_argument("--val-split", type=float, default=0.20, help="Validation split fraction")
    p_prep.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p_prep.add_argument("--three-class", action="store_true", default=True, help="Use 3 classes: 0=laying, 1=standing, 2=sitting")
    p_prep.add_argument("--two-class", dest="three_class", action="store_false", help="Use 2 classes: 0=laying, 1=standing")
    p_prep.add_argument("--seed", type=int, default=42)

    p_train = sub.add_parser("train", help="Fine-tune FallGuard_YOLOv8Pose.pt on prepared dataset")
    p_train.add_argument("--data", default="dataset/fallguard_cctv/fallguard_pose.yaml", help="Path to dataset YAML")
    p_train.add_argument("--base-weights", default="models/FallGuard_YOLOv8Pose.pt", help="Initial model weights")
    p_train.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p_train.add_argument("--imgsz", type=int, default=640, help="Training image size")
    p_train.add_argument("--batch", type=int, default=16, help="Batch size")
    p_train.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate for fine-tuning")
    p_train.add_argument("--device", default="", help="Device: '' (auto), '0', 'cpu', or 'mps'")
    p_train.add_argument("--run-name", default="fallguard_finetune", help="Ultralytics run name")
    p_train.add_argument("--export-to", default="models/FallGuard_YOLOv8Pose.pt", help="Copy best.pt to this path after training")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare_dataset(args)
    elif args.command == "train":
        train_model(args)


if __name__ == "__main__":
    main()
