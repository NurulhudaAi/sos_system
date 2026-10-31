#!/usr/bin/env python3
"""
eval/evaluate_dorm_dataset.py — Automated Benchmark & Metric Evaluator for Recorded Dorm Datasets
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compares system detection output against ground truth labeled events:
1. Matches detected Fall, Hand SOS, and Pose SOS alerts against time-windowed ground truth events.
2. Accurately flags:
   - True Positives (TP): correct alert fired during the expected window.
   - False Positives (FP): alert fired during normal/resting/squatting/sitting.
   - False Negatives (FN): missed event (e.g. real fall not detected).
3. Produces a complete Production Readiness Report:
   - Fall Precision, Recall, and F1 Score
   - Hand SOS Precision, Recall, and F1 Score
   - Pose SOS Precision, Recall, and F1 Score
   - False Alarm Count per Hour
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluator")


def evaluate_dataset(video_file: Path, label_file: Path, out_dir: Path, skip: int = 2):
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_report_dir = out_dir / "scan_output"

    labels_data = json.loads(label_file.read_text())
    events = labels_data.get("events", [])
    duration = labels_data.get("duration_seconds", 0.0)

    logger.info(f"Loaded {len(events)} ground-truth events from {label_file.name} (Duration: {duration:.1f}s)")

    # Run discovery scan over the recorded video
    scan_cmd = [
        sys.executable,
        "eval/scan_today_dataset.py",
        "--video", video_file.name,
        "--videos-dir", str(video_file.parent),
        "--out-dir", str(scan_report_dir),
        "--skip", str(skip),
    ]

    logger.info(f"Running detector pipeline on {video_file.name}...")
    subprocess.run(scan_cmd, check=True)

    summary_json_path = scan_report_dir / "scan_summary.json"
    if not summary_json_path.exists():
        logger.error(f"Scan summary not found at {summary_json_path}")
        return

    detected_summary = json.loads(summary_json_path.read_text())
    detected_alerts = []
    for vid_info in detected_summary.get("videos", []):
        detected_alerts.extend(vid_info.get("events", []))

    logger.info(f"Pipeline finished: {len(detected_alerts)} total alerts emitted.")

    # Match Ground Truth against Detected Alerts
    # Tolerance window: alert can occur within event duration + 10s (alert dispatch delay)
    metrics = {
        "fall": {"gt_count": 0, "tp": 0, "fn": 0, "fp": 0},
        "hand_sos": {"gt_count": 0, "tp": 0, "fn": 0, "fp": 0},
        "pose_sos": {"gt_count": 0, "tp": 0, "fn": 0, "fp": 0},
        "false_alarms_by_activity": {
            "resting_bench": 0,
            "sitting_chair": 0,
            "squatting": 0,
            "walking_normal": 0,
            "idle": 0,
        }
    }

    # Count Ground Truth
    gt_fall = [e for e in events if e["label"] == "fall_floor"]
    gt_hand = [e for e in events if e["label"] == "hand_sos"]
    gt_pose = [e for e in events if e["label"] == "pose_sos"]

    metrics["fall"]["gt_count"] = len(gt_fall)
    metrics["hand_sos"]["gt_count"] = len(gt_hand)
    metrics["pose_sos"]["gt_count"] = len(gt_pose)

    matched_alerts = set()

    # Check Falls
    for gt in gt_fall:
        t_start, t_end = gt["start"], gt["end"] + 12.0
        hit = False
        for i, alert in enumerate(detected_alerts):
            if alert.get("type") == "Fall" and (t_start <= alert.get("time", 0.0) <= t_end):
                hit = True
                matched_alerts.add(i)
                break
        if hit:
            metrics["fall"]["tp"] += 1
        else:
            metrics["fall"]["fn"] += 1

    # Check Hand SOS
    for gt in gt_hand:
        t_start, t_end = gt["start"] - 2.0, gt["end"] + 6.0
        hit = False
        for i, alert in enumerate(detected_alerts):
            if alert.get("type") == "Gesture" and "Hand" in alert.get("label", ""):
                if t_start <= alert.get("time", 0.0) <= t_end:
                    hit = True
                    matched_alerts.add(i)
                    break
        if hit:
            metrics["hand_sos"]["tp"] += 1
        else:
            metrics["hand_sos"]["fn"] += 1

    # Check Pose SOS
    for gt in gt_pose:
        t_start, t_end = gt["start"] - 2.0, gt["end"] + 10.0
        hit = False
        for i, alert in enumerate(detected_alerts):
            if alert.get("type") == "Gesture" and ("Overhead" in alert.get("label", "") or "Triangle" in alert.get("label", "")):
                if t_start <= alert.get("time", 0.0) <= t_end:
                    hit = True
                    matched_alerts.add(i)
                    break
        if hit:
            metrics["pose_sos"]["tp"] += 1
        else:
            metrics["pose_sos"]["fn"] += 1

    # Categorize Unmatched Alerts as False Positives
    for i, alert in enumerate(detected_alerts):
        if i not in matched_alerts:
            a_type = alert.get("type", "").lower()
            a_time = alert.get("time", 0.0)
            if a_type == "fall":
                metrics["fall"]["fp"] += 1
            else:
                metrics["hand_sos"]["fp"] += 1

            # Find what activity was actually happening during this False Positive
            activity_label = "idle"
            for e in events:
                if e["start"] <= a_time <= e["end"]:
                    activity_label = e["label"]
                    break
            metrics["false_alarms_by_activity"][activity_label] = metrics["false_alarms_by_activity"].get(activity_label, 0) + 1

    # Calculate Rates
    def calc_pr(m):
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if fn == 0 else 0.0)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return round(prec * 100, 1), round(rec * 100, 1), round(f1 * 100, 1)

    fall_p, fall_r, fall_f1 = calc_pr(metrics["fall"])
    hand_p, hand_r, hand_f1 = calc_pr(metrics["hand_sos"])
    pose_p, pose_r, pose_f1 = calc_pr(metrics["pose_sos"])

    hours = max(1e-4, duration / 3600.0)
    total_fps = metrics["fall"]["fp"] + metrics["hand_sos"]["fp"] + metrics["pose_sos"]["fp"]
    fa_per_hour = round(total_fps / hours, 2)

    # Render Markdown Report
    report = f"""# Dormitory CCTV Production Evaluation Report
**Session Video:** `{video_file.name}`  
**Duration:** {duration:.1f}s ({round(duration/60, 1)} minutes)  
**Total Alerts:** {len(detected_alerts)}  
**Total False Alarms:** {total_fps} ({fa_per_hour} FA / hour)  

---

## 📊 Accuracy & Reliability Metrics

| Signal / Detector | Ground Truth Events | True Positives (TP) | False Positives (FP) | Missed (FN) | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|
| **Fall Detection** | {metrics["fall"]["gt_count"]} | {metrics["fall"]["tp"]} | {metrics["fall"]["fp"]} | {metrics["fall"]["fn"]} | **{fall_p}%** | **{fall_r}%** | **{fall_f1}%** |
| **Hand SOS (MediaPipe)** | {metrics["hand_sos"]["gt_count"]} | {metrics["hand_sos"]["tp"]} | {metrics["hand_sos"]["fp"]} | {metrics["hand_sos"]["fn"]} | **{hand_p}%** | **{hand_r}%** | **{hand_f1}%** |
| **Pose SOS (Overhead)** | {metrics["pose_sos"]["gt_count"]} | {metrics["pose_sos"]["tp"]} | {metrics["pose_sos"]["fp"]} | {metrics["pose_sos"]["fn"]} | **{pose_p}%** | **{pose_r}%** | **{pose_f1}%** |

---

## 🛡️ False Alarm Breakdown by Normal Activities

| Activity | False Alarms Recorded | Target for Production |
|---|---|---|
| 🛋️ **Resting on Bench** | {metrics["false_alarms_by_activity"].get("resting_bench", 0)} | 0 (Must be 0) |
| 🪑 **Sitting on Chair** | {metrics["false_alarms_by_activity"].get("sitting_chair", 0)} | 0 (Must be 0) |
| 🧊 **Squatting / Crouching** | {metrics["false_alarms_by_activity"].get("squatting", 0)} | 0 (Must be 0) |
| 🚶 **Normal Walking (Back/Front)** | {metrics["false_alarms_by_activity"].get("walking_normal", 0)} | 0 (Must be 0) |
| ⚪ **Idle / Empty Room** | {metrics["false_alarms_by_activity"].get("idle", 0)} | 0 (Must be 0) |

---

## 🎯 Production Gate Verdict
"""
    passed = (fall_r >= 95.0 or metrics["fall"]["gt_count"] == 0) and total_fps == 0
    if passed:
        report += "\n### ✅ PASSED PRODUCTION GATE (Ready for 24/7 Deployment)\n"
    else:
        report += f"\n### ⚠️ ACTION REQUIRED: {total_fps} False Positives / {metrics['fall']['fn']} Missed Falls detected.\n"

    report_path = out_dir / "evaluation_report.md"
    report_path.write_text(report)
    print("\n" + report)
    logger.info(f"✅ Report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate recorded dorm session against ground truth")
    parser.add_argument("--video", "-v", type=str, required=True, help="Path to recorded .mp4 video")
    parser.add_argument("--labels", "-l", type=str, default="", help="Path to ground truth labels JSON (default: same stem + _labels.json)")
    parser.add_argument("--out-dir", "-o", type=str, default="eval/dorm_eval_report", help="Output directory")
    parser.add_argument("--skip", type=int, default=2, help="Frame skip")
    args = parser.parse_args()

    v_path = Path(args.video)
    l_path = Path(args.labels) if args.labels else v_path.parent / f"{v_path.stem}_labels.json"

    if not v_path.exists():
        logger.error(f"Video file not found: {v_path}")
        sys.exit(1)
    if not l_path.exists():
        logger.error(f"Label file not found: {l_path}")
        sys.exit(1)

    evaluate_dataset(v_path, l_path, Path(args.out_dir), skip=args.skip)
