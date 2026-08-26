#!/usr/bin/env python3
"""
eval/inference_only.py — Inference-Only Detection (No Labels Required)

รันโมเดล fall + hand_sos บนวิดีโอโดยไม่ต้องมี label
Output:
  - ตาราง predictions ทั้งหมด (console + CSV)
  - Debug video พร้อม skeleton/bbox/hand overlay
  - (Optional) สร้างไฟล์ label จากผล predictions

Usage:
    python3 model_server.py &
    python3 eval/inference_only.py \
        --videos-dir /path/to/videos \
        --detector-url http://127.0.0.1:8000 \
        --out-dir eval/report_inference \
        --glob "rtsp[0-9]*" \
        --save-debug-video

    # สร้าง label จาก predictions:
    python3 eval/inference_only.py \
        --videos-dir /path/to/videos \
        --glob "rtsp[0-9]*" \
        --generate-labels
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
from detectors.hands_over_head_detector import HandsOverHeadDetector

# Import shared drawing functions from eval_harness_full
from eval.eval_harness_full import (
    SimpleTracker,
    detect_all,
    apply_clahe,
    draw_skeleton,
    draw_bbox,
    draw_hud,
    SKELETON_CONNECTIONS,
)


def inference_video(video_path: Path, detector_url: str,
                    fall_cfg: dict, hand_cfg: dict,
                    head_cfg: dict = None,
                    sample_fps: int = 5,
                    save_debug_video: bool = False,
                    debug_video_dir: Path = None) -> Tuple[List[dict], float]:
    """Process one video — inference only, no GT matching."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️  cannot open {video_path}")
        return [], 0.0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / src_fps if src_fps > 0 else 0
    frame_interval = max(1, round(src_fps / sample_fps))

    print(f"  📐 {frame_w}×{frame_h} @ {src_fps:.1f}fps, "
          f"{total_frames} frames ({duration:.1f}s)")

    fall_d = FallDetector(fall_cfg)
    hand_d = HandSOSDetector(hand_cfg)
    head_d = HandsOverHeadDetector(head_cfg or {})
    tracker = SimpleTracker()

    # Per-track state (same as main.py & eval_harness_full)
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
    min_consec_sos = max(1, int(hand_cfg.get("min_consec_sos_frames", 3)))
    hand_consec_count: dict = {}

    cooldown_fall = {}
    cooldown_hand = {}
    cooldown_pose = {}
    hand_last_event_sec = -1e9
    COOLDOWN_SEC = fall_cfg.get("cooldown_seconds", 120)
    HAND_COOLDOWN = hand_cfg.get("cooldown_seconds", 60)
    POSE_COOLDOWN = (head_cfg or {}).get("cooldown_seconds", 60)

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
    last_progress = -1

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

        # Progress bar (every 10%)
        progress = int(frame_idx / total_frames * 10) if total_frames > 0 else 0
        if progress != last_progress:
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            bar = "█" * progress + "░" * (10 - progress)
            print(f"\r  [{bar}] {pct:.0f}% (t={t_sec:.1f}s, "
                  f"{len(predictions)} preds)", end="", flush=True)
            last_progress = progress

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

        # ── CLAHE for hand detection ──
        frame_clahe = apply_clahe(frame)

        # ── Per-person processing ──
        for i, det in enumerate(assigned):
            tid = det["track_id"]
            bbox = det["bbox"]
            kps = people_raw[i].get("keypoints", []) if i < len(people_raw) else []
            conf = people_raw[i].get("conf", 0.0) if i < len(people_raw) else 0.0

            # ── Fall Detection (pass video timestamp) ──
            fr = None
            try:
                fr = fall_d.process(tid, kps, bbox, h, w, timestamp=t_sec)
                fall_states[tid] = fr
            except Exception:
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
                    "details": {
                        "spine_angle": round(fr.get("spine_angle", 0), 1),
                        "bbox_ratio": round(fr.get("bbox_ratio", 0), 2),
                        "is_critical": bool(is_critical),
                    },
                })
                print(f"\n  🚨 FALL @ t={t_sec:.1f}s (T{tid}, "
                      f"angle={fr.get('spine_angle', 0):.0f}°, "
                      f"ratio={fr.get('bbox_ratio', 0):.1f})")
            elif not is_confirmed and tid in cooldown_fall:
                if t_sec - cooldown_fall[tid] > COOLDOWN_SEC:
                    del cooldown_fall[tid]

            # ── Pose SOS Detection (hands over head) ──
            try:
                pose_r = head_d.process(tid, kps, h, w, timestamp=t_sec)
                if pose_r.get("triggered"):
                    last_pose = cooldown_pose.get(tid, -1e9)
                    if t_sec - last_pose >= POSE_COOLDOWN:
                        cooldown_pose[tid] = t_sec
                        predictions.append({
                            "video": video_path.name,
                            "event_type": "pose_sos",
                            "track_id": tid,
                            "t_sec": round(t_sec, 2),
                            "confidence": 1.0,
                            "details": {"time_held": pose_r.get("time_held", 0)},
                        })
                        print(f"\n  🙌 POSE_SOS @ t={t_sec:.1f}s (T{tid}, "
                              f"held={pose_r.get('time_held', 0):.1f}s)")
            except Exception:
                pass

            # ── Hand SOS Detection ──
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

            # Expand bbox by 10%
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

                        # Draw hand landmarks on debug frame
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
            temporal_confirmed = (len(recent_sos) >= hand_temporal_hits_required and
                                  sos_hits >= hand_temporal_hits_required)
            fast_sos_confirmed = is_sos_frame and prev_hs >= 1
            sos_confirmed = temporal_confirmed or fast_sos_confirmed

            # [FIX-FP] Consecutive SOS frames gate
            if is_sos_frame:
                hand_consec_count[tid] = hand_consec_count.get(tid, 0) + 1
            else:
                hand_consec_count[tid] = 0
            consec_ok = hand_consec_count.get(tid, 0) >= min_consec_sos

            if sos_confirmed and consec_ok:
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
                        "details": {"hand_state": hs},
                    })
                    print(f"\n  🆘 HAND_SOS @ t={t_sec:.1f}s (T{tid})")

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
    print(f"\n  ✅ {video_path.name}: {len(predictions)} predictions "
          f"in {elapsed:.1f}s (video={video_seconds:.1f}s, "
          f"speed={video_seconds/elapsed:.1f}x)")
    return predictions, video_seconds


def generate_labels_from_predictions(predictions: List[dict],
                                      out_dir: Path,
                                      margin_sec: float = 2.0):
    """Generate label JSON files from predictions for later eval."""
    by_video = defaultdict(list)
    for p in predictions:
        by_video[p["video"]].append(p)

    labels_dir = out_dir / "generated_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    for video_name, preds in by_video.items():
        stem = Path(video_name).stem
        events = []
        for p in preds:
            events.append({
                "event_type": p["event_type"],
                "start_sec": max(0, p["t_sec"] - margin_sec),
                "end_sec": p["t_sec"] + margin_sec,
                "notes": f"auto-generated from inference (T{p['track_id']})",
            })

        label = {
            "video": video_name,
            "location": "RTSP Camera",
            "events": events,
        }
        label_path = labels_dir / f"{stem}.json"
        label_path.write_text(
            json.dumps(label, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  📝 Label → {label_path} ({len(events)} events)")

    return labels_dir


def main():
    ap = argparse.ArgumentParser(
        description="Inference-Only Detection — No Labels Required"
    )
    ap.add_argument("--videos-dir", required=True,
                    help="Directory containing video files")
    ap.add_argument("--detector-url", default="http://127.0.0.1:8000")
    ap.add_argument("--out-dir", default="eval/report_inference")
    ap.add_argument("--sample-fps", type=int, default=5)
    ap.add_argument("--save-debug-video", action="store_true",
                    help="Save debug videos with skeleton/bbox overlay")
    ap.add_argument("--glob", default="rtsp[0-9]*",
                    help="Glob pattern for video filenames (default: rtsp[0-9]*)")
    ap.add_argument("--generate-labels", action="store_true",
                    help="Generate label files from predictions")
    ap.add_argument("--label-margin", type=float, default=2.0,
                    help="Margin in seconds for generated label windows")
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
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
    head_cfg = thresholds.get("hands_over_head", {})

    # Find video files matching glob
    import fnmatch
    video_files = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in ('.mp4', '.avi', '.mkv', '.mov')
        and fnmatch.fnmatch(p.stem, args.glob)
    )

    if not video_files:
        print(f"⚠️  No videos matching '{args.glob}' in {videos_dir}")
        print(f"   Available files:")
        for f in sorted(videos_dir.iterdir()):
            if f.suffix.lower() in ('.mp4', '.avi', '.mkv', '.mov'):
                print(f"     {f.name} ({f.stat().st_size / 1e6:.1f}MB)")
        return

    print(f"\n{'='*70}")
    print(f"🔍 Inference-Only Detection (No Labels)")
    print(f"{'='*70}")
    print(f"Videos: {len(video_files)} files")
    print(f"Detector: {args.detector_url}")
    print(f"Sample FPS: {args.sample_fps}")
    print(f"Debug video: {'✅' if args.save_debug_video else '❌'}")
    print(f"{'='*70}\n")

    for i, vf in enumerate(video_files, 1):
        mb = vf.stat().st_size / 1e6
        print(f"  {i}. {vf.name} ({mb:.1f}MB)")
    print()

    all_predictions = []
    total_video_seconds = 0.0

    for vf in video_files:
        print(f"\n▶ {vf.name}")
        preds, vid_secs = inference_video(
            vf, args.detector_url,
            fall_cfg, hand_cfg, head_cfg,
            sample_fps=args.sample_fps,
            save_debug_video=args.save_debug_video,
            debug_video_dir=debug_video_dir,
        )
        all_predictions.extend(preds)
        total_video_seconds += vid_secs

    # ── Save CSV ──
    csv_path = out_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video", "event_type", "track_id", "t_sec", "confidence",
        ])
        writer.writeheader()
        for p in all_predictions:
            writer.writerow({
                "video": p["video"],
                "event_type": p["event_type"],
                "track_id": p["track_id"],
                "t_sec": p["t_sec"],
                "confidence": p["confidence"],
            })

    # ── Save full JSON ──
    json_path = out_dir / "predictions.json"
    json_path.write_text(
        json.dumps(all_predictions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Print Summary Table ──
    hours = total_video_seconds / 3600.0
    print(f"\n{'='*70}")
    print(f"📊 INFERENCE RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total video time: {total_video_seconds:.1f}s ({hours:.4f}hr)")
    print(f"Total predictions: {len(all_predictions)}")
    print()

    # Group by video + event_type
    by_video = defaultdict(list)
    for p in all_predictions:
        by_video[p["video"]].append(p)

    print(f"{'Video':<25}{'Event':<12}{'Time(s)':<10}{'T_ID':<6}{'Conf':<8}")
    print(f"{'-'*70}")
    for vname in sorted(by_video.keys()):
        preds = sorted(by_video[vname], key=lambda x: x["t_sec"])
        for p in preds:
            emojis = {"hand_sos": "🆘", "fall": "🚨", "pose_sos": "🙌"}
            emoji = emojis.get(p["event_type"], "❓")
            print(f"{emoji} {vname:<23}{p['event_type']:<12}"
                  f"{p['t_sec']:<10.1f}{p['track_id']:<6}{p['confidence']:<8.2f}")
        if not preds:
            print(f"  {vname:<23}(no detections)")

    # Count by type
    type_counts = defaultdict(int)
    for p in all_predictions:
        type_counts[p["event_type"]] += 1

    print(f"\n{'-'*70}")
    print(f"Event counts:")
    for et, cnt in sorted(type_counts.items()):
        print(f"  {et}: {cnt}")

    # Videos with no detections
    no_det_videos = [vf.name for vf in video_files if vf.name not in by_video]
    if no_det_videos:
        print(f"\n⚠️  Videos with NO detections: {', '.join(no_det_videos)}")

    print(f"\n📄 Predictions CSV: {csv_path}")
    print(f"📄 Predictions JSON: {json_path}")
    if args.save_debug_video:
        print(f"🎬 Debug videos: {debug_video_dir}/")

    # ── Generate Labels (optional) ──
    if args.generate_labels and all_predictions:
        print(f"\n{'='*70}")
        print(f"📝 Generating label files from predictions...")
        labels_dir = generate_labels_from_predictions(
            all_predictions, out_dir, args.label_margin
        )
        print(f"\n✅ Labels saved to: {labels_dir}/")
        print(f"   Copy to eval/labels/ to use with eval_harness_full.py")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
