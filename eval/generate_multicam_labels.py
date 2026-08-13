#!/usr/bin/env python3
"""
eval/generate_multicam_labels.py — Convert MultiCam Fall Dataset annotations
(data_tuple3.csv) into JSON labels for eval_augmented.py / eval_harness.py.

Usage:
    python3 eval/generate_multicam_labels.py \
        --dataset-dir /Users/nurulhudaadamishaq/Downloads/dataset \
        --cam 1 \
        --out-labels-dir eval/labels \
        --out-videos-dir eval/videos
"""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

import cv2


def read_annotations(csv_path: Path) -> Dict[Tuple[int, int], List[dict]]:
    """Read data_tuple3.csv -> dict keyed by (chute, cam)."""
    data = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chute = int(float(row["chute"]))
            cam = int(float(row["cam"]))
            start = int(float(row["start"]))
            end = int(float(row["end"]))
            label = int(float(row["label"]))
            data[(chute, cam)].append({
                "start_frame": start,
                "end_frame": end,
                "label": label,
            })
    return data


def merge_fall_segments(segments: List[dict], gap_frames: int = 5) -> List[dict]:
    """Merge consecutive or overlapping fall (label=1) segments."""
    falls = sorted(
        [s for s in segments if s["label"] == 1],
        key=lambda x: x["start_frame"],
    )
    if not falls:
        return []

    merged = []
    current = {"start_frame": falls[0]["start_frame"], "end_frame": falls[0]["end_frame"]}

    for seg in falls[1:]:
        if seg["start_frame"] <= current["end_frame"] + gap_frames:
            current["end_frame"] = max(current["end_frame"], seg["end_frame"])
        else:
            merged.append(current)
            current = {"start_frame": seg["start_frame"], "end_frame": seg["end_frame"]}

    merged.append(current)
    return merged


def get_video_fps(video_path: Path) -> float:
    """Get FPS from video file, fallback to 120."""
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
        cap.release()
        return fps
    return 120.0


def generate_label(chute, cam, segments, fps, video_filename):
    """Generate a single JSON label dict from annotation segments."""
    merged = merge_fall_segments(segments)

    events = []
    for m in merged:
        events.append({
            "event_type": "fall",
            "start_sec": round(m["start_frame"] / fps, 2),
            "end_sec": round(m["end_frame"] / fps, 2),
            "notes": f"auto-generated from data_tuple3.csv (frames {m['start_frame']}-{m['end_frame']})",
        })

    label = {
        "video": video_filename,
        "location": f"multicam_chute{chute:02d}",
        "events": events,
    }

    non_fall_count = sum(1 for s in segments if s["label"] == 0)
    label["_source"] = (
        f"MultiCam Fall Dataset: chute{chute:02d}/cam{cam}.avi | "
        f"{len(merged)} fall event(s), {non_fall_count} confounding segment(s) | "
        f"FPS={fps}"
    )

    return label


def main():
    ap = argparse.ArgumentParser(
        description="Convert MultiCam Fall Dataset annotations to eval label JSONs"
    )
    ap.add_argument("--dataset-dir", required=True,
                    help="Path to MultiCam dataset root (contains chute01-24/ and data_tuple3.csv)")
    ap.add_argument("--cam", type=int, nargs="+", default=[1],
                    help="Camera IDs to include (default: [1])")
    ap.add_argument("--out-labels-dir", default="eval/labels",
                    help="Output directory for JSON labels")
    ap.add_argument("--out-videos-dir", default="eval/videos",
                    help="Output directory for video symlinks")
    ap.add_argument("--link-mode", choices=["symlink", "copy", "skip"], default="symlink",
                    help="How to handle video files (default: symlink)")
    ap.add_argument("--fps-override", type=float, default=None,
                    help="Override FPS value (default: read from video)")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    csv_path = dataset_dir / "data_tuple3.csv"

    if not csv_path.exists():
        print(f"Cannot find {csv_path}")
        return

    labels_dir = Path(args.out_labels_dir)
    videos_dir = Path(args.out_videos_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    annotations = read_annotations(csv_path)

    all_chutes = set()
    for d in sorted(dataset_dir.iterdir()):
        if d.is_dir() and d.name.startswith("chute"):
            try:
                chute_num = int(d.name.replace("chute", ""))
                all_chutes.add(chute_num)
            except ValueError:
                pass

    print(f"Dataset: {dataset_dir}")
    print(f"Annotation: {csv_path}")
    print(f"Found {len(all_chutes)} scenarios: chute{min(all_chutes):02d}-chute{max(all_chutes):02d}")
    print(f"Camera(s): {args.cam}")
    print("=" * 60)

    created_count = 0
    skipped_count = 0

    for chute in sorted(all_chutes):
        for cam in args.cam:
            video_path = dataset_dir / f"chute{chute:02d}" / f"cam{cam}.avi"

            if not video_path.exists():
                print(f"  Missing: {video_path}")
                skipped_count += 1
                continue

            fps = args.fps_override or get_video_fps(video_path)
            video_filename = f"chute{chute:02d}_cam{cam}.avi"
            key = (chute, cam)
            segments = annotations.get(key, [])

            label = generate_label(chute, cam, segments, fps, video_filename)

            label_filename = f"multicam_chute{chute:02d}_cam{cam}.json"
            label_path = labels_dir / label_filename
            label_path.write_text(
                json.dumps(label, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            video_dest = videos_dir / video_filename
            if args.link_mode == "symlink":
                if video_dest.exists() or video_dest.is_symlink():
                    video_dest.unlink()
                os.symlink(video_path.resolve(), video_dest)
            elif args.link_mode == "copy":
                if not video_dest.exists():
                    shutil.copy2(video_path, video_dest)

            n_fall = len(label["events"])
            n_confounding = sum(1 for s in segments if s["label"] == 0)
            fall_info = ", ".join(
                f"{e['start_sec']:.1f}-{e['end_sec']:.1f}s" for e in label["events"]
            ) if n_fall > 0 else "none"

            status = "FALL" if n_fall > 0 else "negative"
            print(f"  chute{chute:02d}/cam{cam} -> {label_filename} "
                  f"| {status} ({n_fall} events: {fall_info}) "
                  f"| {n_confounding} confounding")

            created_count += 1

    print(f"\n{'='*60}")
    print(f"Created {created_count} label files in {labels_dir}/")
    if skipped_count:
        print(f"Skipped {skipped_count} missing videos")
    print(f"Videos linked in {videos_dir}/")
    print(f"\nNext: run evaluation with:")
    print(f"   python3 eval/eval_augmented.py \\")
    print(f"     --videos-dir {videos_dir} \\")
    print(f"     --labels-dir {labels_dir} \\")
    print(f"     --out-dir eval/report_multicam \\")
    print(f"     --augmentations all")


if __name__ == "__main__":
    main()
