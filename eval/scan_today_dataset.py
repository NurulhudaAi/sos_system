#!/usr/bin/env python3
"""
eval/scan_today_dataset.py — Automated Scan & Discovery for new dataset videos.
Scans videos through all 4 detectors (Fall, Hand SOS, Pose SOS, Object Guardian),
captures every detected event with timestamps and bounding box snapshot images,
and compiles a comprehensive event log.
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime

import cv2
import numpy as np
import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detectors.fall_detector import FallDetector
from detectors.hand_sos_detector import HandSOSDetector
from detectors.hands_over_head_detector import HandsOverHeadDetector
from detectors.object_guardian import ObjectGuardian
from utils import preprocess, save_annotated_snapshot

DET_URL = os.getenv("DETECTOR_URL", "http://127.0.0.1:8000")


def _are_both_arms_over_head(kps, conf_thresh=0.25):
    """Checks if both arms are genuinely raised overhead in a wide V-shape (Pose SOS).
    Requires wrists to be spread apart, preventing Hand SOS or holding objects from being suppressed.
    """
    if len(kps) < 11:
        return False
    l_wr, r_wr = kps[9], kps[10]
    nose = kps[0]
    l_sh, r_sh = kps[5], kps[6]
    if float(l_wr[2]) < conf_thresh or float(r_wr[2]) < conf_thresh:
        return False
    if float(l_sh[2]) < conf_thresh or float(r_sh[2]) < conf_thresh:
        return False

    shoulder_y = min(float(l_sh[1]), float(r_sh[1]))
    shoulder_w = abs(float(l_sh[0]) - float(r_sh[0]))
    wrist_sep = abs(float(l_wr[0]) - float(r_wr[0]))

    # Must have genuine V-shape separation (arms spread apart)
    min_sep = max(18.0, 0.55 * shoulder_w)
    if wrist_sep < min_sep:
        return False

    head_ref = float(nose[1]) if float(nose[2]) >= conf_thresh else (shoulder_y - 15.0)

    # Both wrists must be near/above head AND clearly above shoulders
    return (
        float(l_wr[1]) <= head_ref + 5.0
        and float(r_wr[1]) <= head_ref + 5.0
        and float(l_wr[1]) <= shoulder_y - 12.0
        and float(r_wr[1]) <= shoulder_y - 12.0
    )


from tracker import HungarianTracker, SimpleTracker, SpatialAlertDeduplicator



def api_detect_all(frame, timeout=5):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    try:
        r = requests.post(f"{DET_URL}/detect_all", files={"image": ("f.jpg", buf.tobytes(), "image/jpeg")}, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        pass
    return {"people": [], "objects": []}


def save_snapshot_all_persons(snap_path, frame, all_assigned, alert_tid, alert_label, alert_color):
    """Draws bounding boxes for EVERY person detected in the frame:
    - Neutral people: gray box with [tid=X]
    - Alert person: highlighted with alert_color and label.
    """
    disp = frame.copy()
    # 1. Background layer: other people in scene
    for p in all_assigned:
        p_tid = p.get("track_id")
        if p_tid != alert_tid:
            bx = p.get("bbox", [0, 0, 0, 0])
            bx1, by1, bx2, by2 = [int(v) for v in bx]
            cv2.rectangle(disp, (bx1, by1), (bx2, by2), (180, 180, 180), 2)
            cv2.putText(disp, f"ID:{p_tid}", (bx1, max(15, by1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    # 2. Foreground layer: active alert person
    for p in all_assigned:
        p_tid = p.get("track_id")
        if p_tid == alert_tid:
            bx = p.get("bbox", [0, 0, 0, 0])
            bx1, by1, bx2, by2 = [int(v) for v in bx]
            cv2.rectangle(disp, (bx1, by1), (bx2, by2), alert_color, 3)
            (lw, lh), _ = cv2.getTextSize(alert_label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            bg_y1 = max(0, by1 - lh - 10)
            bg_y2 = by1
            cv2.rectangle(disp, (bx1, bg_y1), (min(disp.shape[1] - 1, bx1 + lw + 10), bg_y2), alert_color, -1)
            cv2.putText(disp, alert_label, (bx1 + 5, bg_y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            break
    cv2.imwrite(str(snap_path), disp)


def scan_single_video(video_path: Path, out_dir: Path, skip_frames: int = 1) -> list:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open {video_path.name}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'='*70}")
    print(f"🎬 Scanning: {video_path.name}")
    print(f"   Resolution: {w_orig}x{h_orig} | FPS: {fps:.1f} | Duration: {duration_s:.1f}s ({total_frames} frames)")
    print(f"{'='*70}")

    cfg = yaml.safe_load((ROOT / "config/thresholds.yaml").read_text())
    hand_cfg = cfg.get("hand_sos", {})
    fall_cfg = cfg.get("fall", {})
    pose_cfg = cfg.get("hands_over_head", {})
    obj_cfg = cfg.get("object_guardian", {})

    W, H = 1920, 1080
    tracker = HungarianTracker(frame_w=W, frame_h=H)
    spatial_dedup = SpatialAlertDeduplicator(radius_norm=0.12, default_cooldown_sec=30.0)
    fall_d = FallDetector(fall_cfg)
    hand_d = HandSOSDetector(hand_cfg)
    head_d = HandsOverHeadDetector(pose_cfg)
    obj_grd = ObjectGuardian({**obj_cfg, "alert_dir": str(out_dir / "snapshots")})

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Tracking states
    hand_states = {}
    hand_miss = {}
    hand_first_seen = {}
    hand_consec3 = {}
    hand_recent_sos = defaultdict(lambda: deque(maxlen=int(hand_cfg.get("temporal_window", 10))))
    _hand_last_ts = {}
    _pose_last_ts = {}
    _fall_last_ts = {}
    fall_ev = {}
    hand_step_time = {}
    track_prev_center = {}

    MIN_CONSEC3 = int(hand_cfg.get("min_consec_sos_frames", 2))
    hand_min_bbox_area = float(hand_cfg.get("min_hand_bbox_area_norm", 0.002))
    hand_cooldown = float(hand_cfg.get("cooldown_seconds", 30))
    pose_cooldown = float(pose_cfg.get("cooldown_seconds", 60))
    fall_cooldown = float(fall_cfg.get("cooldown_seconds", 120))

    events = []
    frame_idx = 0
    t_start = time.time()

    snap_dir = out_dir / "snapshots" / video_path.stem
    snap_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ret, raw = cap.read()
        if not ret:
            break
        frame_idx += 1
        if skip_frames > 1 and frame_idx % skip_frames != 0:
            continue

        frame_ts = frame_idx / fps
        frame = preprocess(raw, W, H)
        clean_frame = frame.copy()
        h, w = frame.shape[:2]

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        frame_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        resp = api_detect_all(frame)
        pdets = resp.get("people", [])
        odets = resp.get("objects", [])

        assigned = tracker.update(pdets) if pdets else []

        # Object Guardian
        for oa in obj_grd.update(clean_frame, odets, assigned, source_id=video_path.stem, t_sec=frame_ts):
            ev_item = {
                "video": video_path.name,
                "timestamp_sec": round(frame_ts, 2),
                "frame_idx": frame_idx,
                "event_type": oa.get("event_type", "object_event"),
                "label": f"{oa.get('event_type').upper()}: {oa.get('class_name')}",
                "confidence": round(float(oa.get("confidence", 0.0)), 2),
                "track_id": oa.get("track_id"),
                "person_track_id": oa.get("person_track_id"),
                "bbox": oa.get("bbox"),
                "details": f"unattended={oa.get('seconds_unattended', 0):.1f}s",
            }
            events.append(ev_item)
            print(f"  🔔 [{frame_ts:6.1f}s] {ev_item['label']} (tid={oa.get('track_id')})")

        # Cleanup lost tracks
        alive = set(tracker.tracks.keys())
        dropped = (set(hand_states.keys()) | set(head_d._since.keys())) - alive
        for tid in dropped:
            fall_d.cleanup_track(tid)
            head_d.cleanup_track(tid)
            for d in [hand_states, hand_miss, hand_consec3, _hand_last_ts, _pose_last_ts, hand_step_time, hand_first_seen]:
                d.pop(tid, None)
            hand_recent_sos.pop(tid, None)

        # People processing
        for i, det in enumerate(assigned):
            tid = det.get("track_id")
            bbox = det.get("bbox", [0, 0, 0, 0])
            conf = float(det.get("conf", 0.0))
            kps = np.array(det.get("keypoints", []))
            posture_class = det.get("posture_class")
            posture_conf = float(det.get("posture_conf", 0.0))

            cx = (bbox[0] + bbox[2]) / 2.0 / w
            cy = (bbox[1] + bbox[3]) / 2.0 / h

            # 1. Fall Detection
            fr = fall_d.process(tid, kps, bbox, h, w, posture_class=posture_class, posture_conf=posture_conf, timestamp=frame_ts)
            is_critical = fr.get("is_critical") and not fr.get("recovered_quickly")
            is_confirmed = (fr.get("is_fallen") or fr.get("danger_lying")) and not fr.get("recovered_quickly")

            if (is_critical or is_confirmed):
                prev_f = _fall_last_ts.get(tid, -1e9)
                is_spatial_dup = spatial_dedup.should_suppress("fall", cx, cy, frame_ts, cooldown_sec=fall_cooldown)
                if (frame_ts - prev_f) >= fall_cooldown and not fall_ev.get(tid) and not is_spatial_dup:
                    lbl = f"FALL [tid={tid}] angle={fr.get('spine_angle', 0):.0f}deg"
                    snap_name = f"fall_tid{tid}_{int(frame_ts)}s.jpg"
                    snap_path = snap_dir / snap_name
                    save_snapshot_all_persons(snap_path, clean_frame, assigned, tid, lbl, (0, 0, 255))

                    ev_item = {
                        "video": video_path.name,
                        "timestamp_sec": round(frame_ts, 2),
                        "frame_idx": frame_idx,
                        "event_type": "fall",
                        "label": lbl,
                        "confidence": round(conf, 2),
                        "track_id": tid,
                        "bbox": [int(v) for v in bbox],
                        "snapshot": str(snap_path.relative_to(out_dir)),
                        "details": f"angle={fr.get('spine_angle')} ratio={fr.get('bbox_ratio')} time_lying={fr.get('time_lying')}",
                    }
                    events.append(ev_item)
                    _fall_last_ts[tid] = frame_ts
                    fall_ev[tid] = True
                    print(f"  🚨 [{frame_ts:6.1f}s] {lbl} -> {snap_name}")
            else:
                if not fr.get("is_fallen"):
                    fall_ev[tid] = False

            # 2. Hand SOS Detection
            both_overhead = _are_both_arms_over_head(kps)
            is_lying = fr.get("is_fallen") or (fr.get("spine_angle") and fr.get("spine_angle") < 40.0)

            if not both_overhead and not is_lying and not fall_ev.get(tid):
                prev_hs = hand_states.get(tid, 0)
                hs = prev_hs
                hdet = False
                if tid not in hand_first_seen:
                    hand_first_seen[tid] = frame_ts

                bbox_area = ((bbox[2]-bbox[0]) * (bbox[3]-bbox[1])) / float(w * h)
                try:
                    hand_lms = hand_d.process_wrist_crop(frame_clahe, kps, h, w, bbox=bbox)
                    if hand_lms:
                        hl = hand_lms[0]
                        hdet = True
                        hs = hand_d.check_sos_step(hs, hl)
                except Exception:
                    pass

                if hs != prev_hs:
                    hand_step_time[tid] = frame_ts
                elif hs >= 1 and (frame_ts - hand_step_time.get(tid, frame_ts)) > 3.5:
                    hs = 0
                    hand_consec3[tid] = 0

                if hdet:
                    hand_miss[tid] = 0
                else:
                    hand_miss[tid] = hand_miss.get(tid, 0) + 1
                    if hand_miss[tid] >= 5:
                        hs = max(0, hs - 1)
                        hand_miss[tid] = 0

                is_fist = bool(hdet and hs == 3 and not hand_d._palm_open(hl) and hand_d._fingers_closed(hl))
                if is_fist:
                    hand_consec3[tid] = hand_consec3.get(tid, 0) + 1
                else:
                    hand_consec3[tid] = 0

                consec_ok = hand_consec3.get(tid, 0) >= MIN_CONSEC3
                is_sos = bool(hdet and hs == 3 and consec_ok)
                hand_recent_sos[tid].append(is_sos)
                sos_hits = sum(1 for ok in hand_recent_sos[tid] if ok)

                if sos_hits >= 3:
                    if (frame_ts - _hand_last_ts.get(tid, -1e9)) >= hand_cooldown:
                        lbl = f"HAND SOS [tid={tid}]"
                        snap_name = f"hand_sos_tid{tid}_{int(frame_ts)}s.jpg"
                        snap_path = snap_dir / snap_name
                        save_snapshot_all_persons(snap_path, clean_frame, assigned, tid, lbl, (0, 255, 0))

                        ev_item = {
                            "video": video_path.name,
                            "timestamp_sec": round(frame_ts, 2),
                            "frame_idx": frame_idx,
                            "event_type": "hand_sos",
                            "label": lbl,
                            "confidence": round(conf, 2),
                            "track_id": tid,
                            "bbox": [int(v) for v in bbox],
                            "snapshot": str(snap_path.relative_to(out_dir)),
                            "details": f"step3_consec={hand_consec3.get(tid)}",
                        }
                        events.append(ev_item)
                        _hand_last_ts[tid] = frame_ts
                        hand_states[tid] = 0
                        hand_consec3[tid] = 0
                        hand_recent_sos[tid].clear()
                        print(f"  🖐️ [{frame_ts:6.1f}s] {lbl} -> {snap_name}")

            # 3. Pose SOS Detection
            if not fall_ev.get(tid) and not is_lying:
                head_res = head_d.process(tid, kps, h, w, timestamp=frame_ts)
                if head_res.get("triggered"):
                    is_spatial_pose_dup = spatial_dedup.should_suppress("pose_sos", cx, cy, frame_ts, cooldown_sec=pose_cooldown)
                    if (frame_ts - _pose_last_ts.get(tid, -1e9)) >= pose_cooldown and not is_spatial_pose_dup:
                        lbl = f"POSE SOS [tid={tid}] held={head_res.get('time_held'):.1f}s"
                        snap_name = f"pose_sos_tid{tid}_{int(frame_ts)}s.jpg"
                        snap_path = snap_dir / snap_name
                        save_snapshot_all_persons(snap_path, clean_frame, assigned, tid, lbl, (0, 165, 255))

                        ev_item = {
                            "video": video_path.name,
                            "timestamp_sec": round(frame_ts, 2),
                            "frame_idx": frame_idx,
                            "event_type": "pose_sos",
                            "label": lbl,
                            "confidence": round(conf, 2),
                            "track_id": tid,
                            "bbox": [int(v) for v in bbox],
                            "snapshot": str(snap_path.relative_to(out_dir)),
                            "details": f"time_held={head_res.get('time_held'):.1f}s",
                        }
                        events.append(ev_item)
                        _pose_last_ts[tid] = frame_ts
                        print(f"  🙌 [{frame_ts:6.1f}s] {lbl} -> {snap_name}")

        if frame_idx % 100 == 0:
            elapsed = time.time() - t_start
            proc_fps = frame_idx / max(1e-3, elapsed)
            pct = (frame_idx / total_frames) * 100
            print(f"   [{pct:5.1f}%] frame {frame_idx}/{total_frames} ({frame_ts:.1f}s) @ {proc_fps:.1f} FPS | events found: {len(events)}")

    cap.release()
    hand_d.release()
    total_elapsed = time.time() - t_start
    print(f"✅ Completed {video_path.name} in {total_elapsed:.1f}s ({frame_idx/max(1e-3, total_elapsed):.1f} FPS) -> Total events: {len(events)}")
    return events


def main():
    parser = argparse.ArgumentParser(description="Scan dataset videos for detections")
    parser.add_argument("--videos-dir", type=str, default="eval/videos", help="Directory of videos to scan")
    parser.add_argument("--video", type=str, default="", help="Scan single video by name")
    parser.add_argument("--out-dir", type=str, default="eval/scan_reports", help="Output directory for reports & snapshots")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame (default=1)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        target_videos = [Path(args.video)]
    else:
        v_dir = Path(args.videos_dir)
        target_videos = sorted([p for p in v_dir.iterdir() if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]])

    if not target_videos:
        print("❌ No videos found to scan.")
        sys.exit(1)

    print(f"Found {len(target_videos)} video(s) to scan.")
    all_events = {}

    for v in target_videos:
        evs = scan_single_video(v, out_dir=out_dir, skip_frames=args.skip)
        all_events[v.name] = evs

    # Write summary JSON and Markdown
    json_path = out_dir / "scan_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_events, f, indent=2, ensure_ascii=False)

    md_path = out_dir / "scan_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Discovery Scan Report — Today Dataset Videos\n\n")
        f.write(f"Scanned at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| Video | Duration | Events Found | Details |\n")
        f.write("|---|---|---|---|\n")
        total_evs = 0
        for vname, evs in all_events.items():
            total_evs += len(evs)
            ev_types = ", ".join(f"{k}: {v}" for k, v in defaultdict(int, [(e['event_type'], 1) for e in evs]).items()) or "None"
            f.write(f"| **{vname}** | - | {len(evs)} | {ev_types} |\n")
        f.write(f"\n**Total Alerts Detected across all videos:** {total_evs}\n\n")

        f.write("## Detailed Event Chronology\n\n")
        for vname, evs in all_events.items():
            f.write(f"### 📹 {vname} ({len(evs)} events)\n\n")
            if not evs:
                f.write("*No alerts triggered (clean pass)*\n\n")
                continue
            f.write("| Time (s) | Type | Label | Track ID | Confidence | Snapshot |\n")
            f.write("|---|---|---|---|---|---|\n")
            for e in evs:
                snap_link = f"[{Path(e['snapshot']).name}]({e['snapshot']})" if e.get("snapshot") else "-"
                f.write(f"| {e['timestamp_sec']}s | `{e['event_type']}` | {e['label']} | {e['track_id']} | {e['confidence']} | {snap_link} |\n")
            f.write("\n")

    print(f"\n🎉 All videos scanned! Report saved to:\n  - {md_path}\n  - {json_path}")


if __name__ == "__main__":
    main()
