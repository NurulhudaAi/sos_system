#!/usr/bin/env python3
import json
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env_manager import init_env
from detectors.fall_detector import FallDetector
from utils import preprocess


def resolve_model_path(model_path):
    p = Path(model_path)
    if p.is_absolute() and p.exists():
        return str(p)
    for candidate in ((ROOT / p).resolve(), (Path.cwd() / p).resolve(), (ROOT / "models" / p.name).resolve()):
        if candidate.exists():
            return str(candidate)
    return model_path


def load_source(source_id):
    sources = yaml.safe_load((ROOT / "config/sources.yaml").read_text()).get("sources", [])
    for source in sources:
        if source.get("id") == source_id or Path(str(source.get("path", ""))).name == source_id:
            return source
    raise SystemExit(f"Source not found in config/sources.yaml: {source_id}")


def json_default(value):
    try:
        import numpy as np

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def main():
    source_id = sys.argv[1] if len(sys.argv) > 1 else "test_cam_1"
    if not init_env(require_edit=False):
        raise SystemExit(1)

    import database

    cfg = yaml.safe_load((ROOT / "config/thresholds.yaml").read_text())
    source = load_source(source_id)
    video_path = Path(source["path"])
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model_path = resolve_model_path(cfg.get("general", {}).get("yolo_model", "../models/yolov8n-pose.pt"))
    yolo = YOLO(model_path)
    fall_d = FallDetector(cfg.get("fall", {}))
    skip = int(cfg.get("general", {}).get("frame_skip", 1))
    conf_thresh = float(cfg.get("general", {}).get("person_conf", 0.30))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = 0
    processed_frames = 0
    detected = False
    detection_frame = None
    detection_video_time = None
    best_conf = -1.0
    best_frame = None
    best_result = None

    start = time.time()
    try:
        while True:
            ret, raw = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % skip != 0:
                continue
            processed_frames += 1
            frame = preprocess(raw, 1920, 1080)
            h, w = frame.shape[:2]

            try:
                results_y = yolo.track(
                    frame,
                    persist=True,
                    conf=conf_thresh,
                    classes=[0],
                    verbose=False,
                    tracker="bytetrack.yaml",
                    device=device,
                )
            except Exception:
                results_y = yolo(frame, conf=conf_thresh, classes=[0], verbose=False, device=device)

            if not results_y or results_y[0].boxes is None:
                continue

            boxes = results_y[0].boxes
            kpts = results_y[0].keypoints
            for i in range(len(boxes)):
                if kpts is None or i >= len(kpts.data):
                    continue
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                det_conf = float(boxes.conf[i].cpu())
                tid = int(boxes.id[i].cpu()) if boxes.id is not None else i
                kp = kpts.data[i].cpu().numpy()
                fr = fall_d.process(tid, kp, bbox, h, w)

                if det_conf > best_conf or fr.get("is_down"):
                    best_conf = det_conf
                    best_frame = raw.copy()
                    best_result = dict(fr, track_id=tid, confidence=round(det_conf, 4), frame=frame_count)

                if fr.get("is_fallen") and not fr.get("recovered_quickly"):
                    detected = True
                    detection_frame = frame_count
                    detection_video_time = round(frame_count / fps, 2) if fps else None
                    best_frame = raw.copy()
                    best_result = dict(fr, track_id=tid, confidence=round(det_conf, 4), frame=frame_count)
                    break
            if detected:
                break
    finally:
        cap.release()

    if best_frame is None:
        cap = cv2.VideoCapture(str(video_path))
        ret, raw = cap.read()
        cap.release()
        if not ret:
            raise SystemExit("No frame available for snapshot")
        best_frame = raw

    now = datetime.now(UTC)
    event_uuid = str(uuid.uuid4())
    event_type = "fall" if detected else "fall_test_no_detection"
    severity = 2 if detected else 0
    severity_name = "HIGH" if detected else "LOG"
    label = "FALL DETECTED" if detected else "FALL TEST - NO CONFIRMED FALL"

    alerts_dir = ROOT / "alerts"
    alerts_dir.mkdir(exist_ok=True)
    ts = now.strftime("%Y%m%d_%H%M%S_%f")
    image_path = alerts_dir / f"{event_type}_{ts}.jpg"
    meta_path = alerts_dir / f"{event_type}_{ts}.json"

    cv2.putText(best_frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    cv2.putText(best_frame, video_path.name, (30, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(str(image_path), best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    meta = {
        "event_uuid": event_uuid,
        "event_type": event_type,
        "detected": detected,
        "source": source,
        "video_path": str(video_path),
        "frame_count": frame_count,
        "processed_frames": processed_frames,
        "detection_frame": detection_frame,
        "detection_video_time_s": detection_video_time,
        "elapsed_s": round(time.time() - start, 2),
        "best_result": best_result or {},
        "image_path": str(image_path),
        "created_at": now.isoformat(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=json_default))

    events_dir = ROOT / "logs/events"
    events_dir.mkdir(parents=True, exist_ok=True)
    event_log = events_dir / f"{now.strftime('%Y-%m-%d')}.jsonl"
    with open(event_log, "a") as f:
        f.write(json.dumps(meta, ensure_ascii=False, default=json_default) + "\n")

    database.insert_sos_event(
        event_uuid=event_uuid,
        event_type=event_type,
        severity=severity,
        severity_name=severity_name,
        source_id=source.get("id"),
        source_path=str(video_path),
        location=source.get("location"),
        track_id=(best_result or {}).get("track_id"),
        image_path=str(image_path),
        meta_path=str(meta_path),
        flags=["FALL_TEST"] if not detected else [],
        extra=meta,
    )

    print(json.dumps({
        "event_uuid": event_uuid,
        "event_type": event_type,
        "detected": detected,
        "image_path": str(image_path),
        "meta_path": str(meta_path),
        "event_log": str(event_log),
        "collection": database.INCIDENTS_COLLECTION,
        "elapsed_s": meta["elapsed_s"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
