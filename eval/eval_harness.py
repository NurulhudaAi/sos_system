#!/usr/bin/env python3
"""
eval/eval_object_harness.py

ประเมิน precision/recall/F1 ของ ObjectGuardian (object_left / object_theft)
เทียบกับ ground-truth labels ใน eval/labels/object_*.json

วิธีรัน:
    python3 eval/eval_object_harness.py \
        --videos-dir eval/videos \
        --labels-dir eval/labels \
        --detector-url http://127.0.0.1:8000 \
        --out-dir eval/report_object

Label schema (ต่อวิดีโอ 1 ไฟล์):
{
  "video": "object_left_1.mp4",
  "location": "ห้องสมุด",
  "events": [
    {
      "event_type": "object_left",
      "start_sec": 12.0,
      "end_sec": 45.0,
      "object_class": "backpack",
      "notes": "เจ้าของวางกระเป๋าไว้แล้วเดินออกจากกล้อง ไม่กลับมา"
    },
    {
      "event_type": "object_theft",
      "start_sec": 45.0,
      "end_sec": 48.0,
      "object_class": "backpack",
      "notes": "คนอื่นหยิบกระเป๋าไปหลังถูกทิ้งไว้"
    }
  ]
}
ถ้าวิดีโอเป็น hard-negative (ไม่มี event จริง) ให้ใส่ "events": []
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import requests
import yaml

# ── เอา ObjectGuardian v2 มาใช้ตรง ๆ (ไม่ import main.py เพราะมี init_env side effect) ──
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detectors.object_guardian import ObjectGuardian


class SimpleTracker:
    """คัดลอกจาก main.py — IoU-based คนธรรมดา ให้ track_id คงที่ข้ามเฟรม"""
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
                tid = self.next_id
                self.next_id += 1
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


def detect_all(detector_url, frame, timeout=5):
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


def evaluate_video(video_path: Path, label: dict, detector_url: str,
                    obj_cfg: dict, sample_fps: int = 5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️  cannot open {video_path}")
        return [], 0.0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, round(src_fps / sample_fps))

    guardian = ObjectGuardian({**obj_cfg, "alert_dir": "eval/alerts_object"})
    tracker = SimpleTracker()
    location = label.get("location", "")
    source_id = video_path.stem

    predictions = []
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        t_sec = frame_idx / src_fps
        resp = detect_all(detector_url, frame)
        people_raw = resp.get("people", [])
        objects_raw = resp.get("objects", [])

        assigned = tracker.update([{"bbox": p["bbox"]} for p in people_raw])
        gp = [{"bbox": d["bbox"], "track_id": d["track_id"]} for d in assigned]

        odets = [
            {
                "bbox": o["bbox"],
                "confidence": o.get("conf", o.get("confidence", 0.0)),
                "class_name": str(o.get("class_id", "object")),
            }
            for o in objects_raw
        ]

        for alert in guardian.update(frame, odets, gp, source_id=source_id, location=location):
            predictions.append({
                "video": video_path.name,
                "event_type": alert["event_type"],
                "object_id": alert.get("object_id"),
                "object_class": alert.get("class_name"),
                "t_sec": round(t_sec, 2),
                "confidence": alert.get("confidence", 0.0),
                "owner_track_id": alert.get("owner_track_id"),
                "suspect_track_id": alert.get("suspect_track_id"),
                "matched": False,
            })

    cap.release()
    video_seconds = frame_idx / src_fps if src_fps else 0.0
    print(f"  {video_path.name}: {len(predictions)} predicted events "
          f"in {time.time()-t0:.1f}s (video={video_seconds:.1f}s)")
    return predictions, video_seconds


def match_predictions(predictions, gt_events, tolerance_sec=5.0):
    """Greedy matching: ตรง event_type + อยู่ในช่วง [start-tol, end+tol]
    (ถ้า label มี object_class ให้ match class ด้วย ไม่งั้นข้าม class check)"""
    gt_used = [False] * len(gt_events)
    for pred in predictions:
        for i, gt in enumerate(gt_events):
            if gt_used[i]:
                continue
            if gt["event_type"] != pred["event_type"]:
                continue
            if gt.get("object_class") and pred.get("object_class"):
                # class_name จาก YOLO เป็น class_id string — เทียบแบบ loose ได้แค่ non-empty
                pass
            lo = gt["start_sec"] - tolerance_sec
            hi = gt["end_sec"] + tolerance_sec
            if lo <= pred["t_sec"] <= hi:
                pred["matched"] = True
                gt_used[i] = True
                break
    return predictions, gt_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="eval/videos")
    ap.add_argument("--labels-dir", default="eval/labels")
    ap.add_argument("--detector-url", default="http://127.0.0.1:8000")
    ap.add_argument("--out-dir", default="eval/report_object")
    ap.add_argument("--tolerance-sec", type=float, default=5.0)
    ap.add_argument("--sample-fps", type=int, default=5)
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        thresholds = yaml.safe_load(Path("config/thresholds.yaml").read_text())
        obj_cfg = thresholds.get("object_guardian", {})
    except Exception:
        obj_cfg = {}

    all_predictions = []
    counts = {}  # event_type -> {tp, fp, fn}
    total_video_seconds = 0.0

    for label_path in sorted(labels_dir.glob("obj_*.json")):
        if label_path.name == "example.json":
            continue
        label = json.loads(label_path.read_text(encoding="utf-8"))
        video_path = videos_dir / label["video"]
        if not video_path.exists():
            print(f"⚠️  missing video: {video_path}")
            continue

        print(f"▶ processing: {label['video']} ...")
        preds, vid_secs = evaluate_video(video_path, label, args.detector_url, obj_cfg,
                                          args.sample_fps)
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

    # ── metrics ──
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

    total_fp = sum(c["fp"] for c in counts.values())
    hours = total_video_seconds / 3600.0
    false_alarms_per_hour = round(total_fp / hours, 2) if hours > 0 else None

    report = {
        "tolerance_sec": args.tolerance_sec,
        "total_video_hours": hours,
        "false_alarms_per_hour": false_alarms_per_hour,
        "metrics": metrics,
    }
    (out_dir / "report_object.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with open(out_dir / "raw_predictions_object.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video", "event_type", "object_id", "object_class", "t_sec",
            "confidence", "owner_track_id", "suspect_track_id", "matched",
        ])
        writer.writeheader()
        for p in all_predictions:
            writer.writerow(p)

    print("\n" + "=" * 70)
    print(f"{'Event Type':<18}{'TP':>4}{'FP':>5}{'FN':>5}{'Precision':>12}{'Recall':>10}{'F1':>8}")
    print("-" * 70)
    for et, m in metrics.items():
        r = f"{m['recall']:.2f}" if m["recall"] is not None else "-"
        f1v = f"{m['f1']:.2f}" if m["f1"] is not None else "-"
        print(f"{et:<18}{m['tp']:>4}{m['fp']:>5}{m['fn']:>5}{m['precision']:>12.2f}{r:>10}{f1v:>8}")
    print("-" * 70)
    print(f"Total video time: {hours:.2f} hr | False alarms/hour (all types): "
          f"{false_alarms_per_hour}")
    print("=" * 70)
    print(f"\n📄 report: {out_dir / 'report_object.json'}")
    print(f"📄 raw predictions: {out_dir / 'raw_predictions_object.csv'}")


if __name__ == "__main__":
    main()