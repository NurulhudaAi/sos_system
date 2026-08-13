#!/usr/bin/env python3
"""
eval/eval_augmented.py — Stress-test the detection pipeline under simulated
real-world CCTV conditions (low resolution, noise, poor lighting, motion blur).

Purpose:
  The main eval_harness tests on clean video. This script applies
  degradation augmentations to simulate conditions that differ between
  test videos and real CCTV deployments, exposing robustness gaps
  BEFORE production.

Usage:
    # Start model_server first:
    python3 model_server.py

    # Then run augmented eval:
    python3 eval/eval_augmented.py \
        --videos-dir eval/videos \
        --labels-dir eval/labels \
        --detector-url http://127.0.0.1:8000 \
        --out-dir eval/report_augmented

    # Run specific augmentations only:
    python3 eval/eval_augmented.py --augmentations low_res dark noisy

    # Adjust severity of augmentations:
    python3 eval/eval_augmented.py --resolution 480 --brightness 0.4 --noise-std 25

Augmentations:
    low_res   — Downscale to --resolution height (default 480px), then upscale back
    dark      — Reduce brightness by --brightness factor (default 0.5)
    noisy     — Add Gaussian noise with --noise-std (default 20)
    blur      — Apply motion blur with --blur-kernel size (default 7)
    all       — Apply all augmentations combined (worst case)
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import requests
import yaml

# ── Import from parent ──
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detectors.fall_detector import FallDetector
from detectors.hand_sos_detector import HandSOSDetector


# ═══════════════════════════════════════════════════════════════════════════════
# Augmentation Functions
# ═══════════════════════════════════════════════════════════════════════════════

def augment_low_res(frame: np.ndarray, target_height: int = 480) -> np.ndarray:
    """Simulate low-resolution CCTV by downscaling then upscaling."""
    h, w = frame.shape[:2]
    if h <= target_height:
        return frame
    scale = target_height / h
    small = cv2.resize(frame, (int(w * scale), target_height), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def augment_dark(frame: np.ndarray, brightness: float = 0.5) -> np.ndarray:
    """Simulate low-light conditions."""
    return cv2.convertScaleAbs(frame, alpha=brightness, beta=0)


def augment_noisy(frame: np.ndarray, std: float = 20.0) -> np.ndarray:
    """Add Gaussian noise to simulate sensor noise in cheap CCTV."""
    noise = np.random.normal(0, std, frame.shape).astype(np.float32)
    noisy = np.clip(frame.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def augment_blur(frame: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Simulate motion blur."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = np.ones(kernel_size) / kernel_size
    return cv2.filter2D(frame, -1, kernel)


def augment_combined(frame: np.ndarray, target_height: int = 480,
                     brightness: float = 0.5, noise_std: float = 20.0,
                     blur_kernel: int = 7) -> np.ndarray:
    """Apply all augmentations — worst-case scenario."""
    frame = augment_low_res(frame, target_height)
    frame = augment_dark(frame, brightness)
    frame = augment_noisy(frame, noise_std)
    frame = augment_blur(frame, blur_kernel)
    return frame


AUGMENTATION_MAP = {
    "low_res": augment_low_res,
    "dark": augment_dark,
    "noisy": augment_noisy,
    "blur": augment_blur,
    "all": augment_combined,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Detection + Tracking (same as main eval harness)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleTracker:
    """IoU-based tracker (copied from eval_harness)."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_video(video_path: Path, label: dict, detector_url: str,
                   fall_cfg: dict, hand_cfg: dict,
                   augment_fn, augment_kwargs: dict,
                   sample_fps: int = 5) -> Tuple[List[dict], float]:
    """Process one video with augmentation, run fall + hand_sos detection."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️  cannot open {video_path}")
        return [], 0.0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, round(src_fps / sample_fps))

    fall_d = FallDetector(fall_cfg)
    hand_d = HandSOSDetector(hand_cfg)
    tracker = SimpleTracker()
    cooldown_fall = {}
    cooldown_hand = {}
    COOLDOWN_SEC = fall_cfg.get("cooldown_seconds", 120)
    HAND_COOLDOWN = hand_cfg.get("cooldown_seconds", 60)

    # Per-track hand state (same as main.py B2)
    hand_states: dict = {}
    hand_miss: dict = {}
    GRACE_FRAMES = 5

    predictions = []
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, raw = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        t_sec = frame_idx / src_fps

        # Apply augmentation
        frame = augment_fn(raw, **augment_kwargs) if augment_kwargs else augment_fn(raw)

        # Detect
        resp = detect_all(detector_url, frame)
        people_raw = resp.get("people", [])

        # Track
        assigned = tracker.update([{"bbox": p["bbox"]} for p in people_raw])
        h, w = frame.shape[:2]

        for i, det in enumerate(assigned):
            tid = det["track_id"]
            bbox = det["bbox"]
            kps = people_raw[i].get("keypoints", []) if i < len(people_raw) else []
            conf = people_raw[i].get("conf", 0.0) if i < len(people_raw) else 0.0

            # ── Fall Detection ──
            fr = fall_d.process(tid, kps, bbox, h, w)
            is_confirmed = (fr.get("is_fallen") or fr.get("danger_lying")) and not fr.get("recovered_quickly")

            if is_confirmed and tid not in cooldown_fall:
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

            # ── Hand SOS Detection ──
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_d.process_frame(frame_rgb)
            if hand_d._results and hand_d._results.hand_landmarks:
                hs = hand_states.get(tid, 0)
                hdet = False
                for hl in hand_d._results.hand_landmarks:
                    try:
                        # Check hand bbox overlap with person bbox
                        hxs = [lm.x * w for lm in hl]
                        hys = [lm.y * h for lm in hl]
                        hcx, hcy = sum(hxs)/len(hxs), sum(hys)/len(hys)
                        if bbox[0] <= hcx <= bbox[2] and bbox[1] <= hcy <= bbox[3]:
                            hdet = True
                            if hs == 0 and hand_d._palm_open(hl):
                                hs = 1
                            elif hs == 1 and hand_d._thumb_in(hl):
                                hs = 2
                            elif hs == 2 and hand_d._fingers_closed(hl):
                                hs = 3
                            break
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

                if hs == 3 and tid not in cooldown_hand:
                    cooldown_hand[tid] = t_sec
                    predictions.append({
                        "video": video_path.name,
                        "event_type": "hand_sos",
                        "track_id": tid,
                        "t_sec": round(t_sec, 2),
                        "confidence": 1.0,
                        "matched": False,
                    })

    cap.release()
    hand_d.release()
    video_seconds = frame_idx / src_fps if src_fps else 0.0
    print(f"  {video_path.name}: {len(predictions)} predictions "
          f"in {time.time()-t0:.1f}s (video={video_seconds:.1f}s)")
    return predictions, video_seconds


def match_predictions(predictions, gt_events, tolerance_sec=5.0):
    """Greedy matching — same logic as main eval harness."""
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
        description="Stress-test SOS detection under simulated CCTV conditions"
    )
    ap.add_argument("--videos-dir", default="eval/videos")
    ap.add_argument("--labels-dir", default="eval/labels")
    ap.add_argument("--detector-url", default="http://127.0.0.1:8000")
    ap.add_argument("--out-dir", default="eval/report_augmented")
    ap.add_argument("--tolerance-sec", type=float, default=5.0)
    ap.add_argument("--sample-fps", type=int, default=5)
    # Augmentation selection
    ap.add_argument("--augmentations", nargs="*",
                    default=["low_res", "dark", "noisy", "blur", "all"],
                    choices=["low_res", "dark", "noisy", "blur", "all"],
                    help="Which augmentations to test (default: all)")
    # Augmentation parameters
    ap.add_argument("--resolution", type=int, default=480,
                    help="Target height for low_res augmentation")
    ap.add_argument("--brightness", type=float, default=0.5,
                    help="Brightness factor for dark augmentation (0-1)")
    ap.add_argument("--noise-std", type=float, default=20.0,
                    help="Gaussian noise std dev")
    ap.add_argument("--blur-kernel", type=int, default=7,
                    help="Motion blur kernel size")
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load thresholds
    try:
        thresholds = yaml.safe_load(Path("config/thresholds.yaml").read_text())
    except Exception:
        thresholds = {}
    fall_cfg = thresholds.get("fall", {})
    hand_cfg = thresholds.get("hand_sos", {})

    # Prepare augmentation configs
    aug_configs = {
        "low_res": (augment_low_res, {"target_height": args.resolution}),
        "dark": (augment_dark, {"brightness": args.brightness}),
        "noisy": (augment_noisy, {"std": args.noise_std}),
        "blur": (augment_blur, {"kernel_size": args.blur_kernel}),
        "all": (augment_combined, {
            "target_height": args.resolution,
            "brightness": args.brightness,
            "noise_std": args.noise_std,
            "blur_kernel": args.blur_kernel,
        }),
    }

    # Collect label files (exclude object labels)
    label_files = sorted(
        p for p in labels_dir.glob("*.json")
        if p.name != "example.json" and not p.name.startswith("obj_")
    )

    all_results = {}

    for aug_name in args.augmentations:
        aug_fn, aug_kwargs = aug_configs[aug_name]
        print(f"\n{'='*60}")
        print(f"🔬 Augmentation: {aug_name}")
        print(f"   Parameters: {aug_kwargs}")
        print(f"{'='*60}")

        all_predictions = []
        counts = {}
        total_video_seconds = 0.0

        for label_path in label_files:
            label = json.loads(label_path.read_text(encoding="utf-8"))
            video_path = videos_dir / label["video"]
            if not video_path.exists():
                print(f"⚠️  missing video: {video_path}")
                continue

            print(f"▶ {label['video']} ...")
            preds, vid_secs = evaluate_video(
                video_path, label, args.detector_url,
                fall_cfg, hand_cfg, aug_fn, aug_kwargs,
                args.sample_fps,
            )
            total_video_seconds += vid_secs

            gt_events = label.get("events", [])
            preds, gt_used = match_predictions(preds, gt_events, args.tolerance_sec)
            all_predictions.extend(preds)

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

        # Metrics
        metrics = compute_metrics(counts)
        total_fp = sum(c["fp"] for c in counts.values())
        hours = total_video_seconds / 3600.0
        fa_rate = round(total_fp / hours, 2) if hours > 0 else None

        report = {
            "augmentation": aug_name,
            "parameters": aug_kwargs,
            "tolerance_sec": args.tolerance_sec,
            "total_video_hours": hours,
            "false_alarms_per_hour": fa_rate,
            "metrics": metrics,
        }

        # Save per-augmentation report
        report_path = out_dir / f"report_{aug_name}.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        csv_path = out_dir / f"raw_predictions_{aug_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "video", "event_type", "track_id", "t_sec", "confidence", "matched",
            ])
            writer.writeheader()
            for p in all_predictions:
                writer.writerow(p)

        all_results[aug_name] = report

        # Print summary
        print(f"\n{'─'*60}")
        print(f"{'Event Type':<18}{'TP':>4}{'FP':>5}{'FN':>5}{'Precision':>12}{'Recall':>10}{'F1':>8}")
        print(f"{'─'*60}")
        for et, m in metrics.items():
            r = f"{m['recall']:.2f}" if m["recall"] is not None else "-"
            f1v = f"{m['f1']:.2f}" if m["f1"] is not None else "-"
            print(f"{et:<18}{m['tp']:>4}{m['fp']:>5}{m['fn']:>5}{m['precision']:>12.2f}{r:>10}{f1v:>8}")
        print(f"{'─'*60}")
        print(f"False alarms/hour: {fa_rate}")

    # ── Comparison Summary ──
    print(f"\n{'='*70}")
    print(f"📊 AUGMENTED EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Augmentation':<15}{'Fall F1':>10}{'Hand F1':>10}{'FA/hr':>10}")
    print(f"{'─'*70}")
    for aug_name, report in all_results.items():
        fall_f1 = report["metrics"].get("fall", {}).get("f1")
        hand_f1 = report["metrics"].get("hand_sos", {}).get("f1")
        fa = report["false_alarms_per_hour"]
        print(f"{aug_name:<15}"
              f"{(f'{fall_f1:.2f}' if fall_f1 is not None else '-'):>10}"
              f"{(f'{hand_f1:.2f}' if hand_f1 is not None else '-'):>10}"
              f"{(f'{fa:.1f}' if fa is not None else '-'):>10}")
    print(f"{'='*70}")
    print(f"\n📄 Reports: {out_dir}/")


if __name__ == "__main__":
    main()
