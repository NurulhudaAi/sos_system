#!/usr/bin/env python3
"""
eval_multi_video.py — Precision/recall/F1 for FALL + HAND_SOS together
on an already-recorded video, when events overlap or a video contains
distractor poses (e.g. hand-over-head that isn't really an SOS call).

Why a combined evaluator (not two separate single-detector runs)
------------------------------------------------------------------
Your production main.py has an explicit per-track rule:

    if sos_confirmed and not fall_ev.get(tid):

...meaning hand_sos is SUPPRESSED for a track while a fall event is
already active on that same track. If you evaluate fall and hand_sos
independently (e.g. with eval_fall_video.py + a hypothetical
eval_hand_sos_video.py run separately), a video where a person raises
their hand right as/after they fall would score hand_sos recall as if
that suppression didn't exist — an inflated number that doesn't match
what the real pipeline will actually log. This script runs BOTH
detectors together, per track, with the SAME suppression rule as
main.py, so the numbers you get here match production behavior.

Ground truth format — supports overlapping/typed events + distractors
-----------------------------------------------------------------------
Each entry is "seconds:type". Recognized types:
    fall        - a real fall happens at this timestamp
    hand_sos    - a real hand-SOS gesture happens at this timestamp
    distractor  - a pose that LOOKS similar but is NOT a real event
                  (e.g. hand over head while stretching, not an SOS
                  call). Any detection landing within tolerance of a
                  distractor timestamp is flagged separately as a
                  "distractor false positive" — useful for catching
                  specificity problems, since it's reported apart from
                  ordinary FPs so you know it wasn't just noise.

Example — a video where the person falls AND raises a hand at ~30s,
plus an unrelated hand-over-head stretch (not SOS) at ~70s:

    python eval_multi_video.py clip.mp4 \\
        --gt "30:fall,31:hand_sos,70:distractor"

Or with a file (one "seconds:type" per line):

    python eval_multi_video.py clip.mp4 --gt-file gt.txt

If you omit ":type" on an entry it defaults to "fall" for backward
compatibility with plain timestamp lists.

Usage
-----
    python eval_multi_video.py clip.mp4 --gt "12:fall,12.5:hand_sos"
    python eval_multi_video.py clip.mp4 --gt-file gt.txt
    python eval_multi_video.py clip.mp4 --no-gt      # counts/timeline only
"""
import argparse
import sys
import time
import json
from pathlib import Path
from collections import defaultdict, deque
from math import ceil

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detectors.fall_detector import FallDetector  # noqa: E402
try:
    from detectors.hand_sos_detector import HandSOSDetector  # noqa: E402
    HAND_SOS_AVAILABLE = True
except Exception as e:
    HAND_SOS_AVAILABLE = False
    _hand_sos_import_error = e


# ── same tracker as test_rtsp_fall.py / eval_fall_video.py ──
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


def load_cfg():
    cfg_path = ROOT / "config" / "thresholds.yaml"
    if cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text())
    print(f"⚠️  {cfg_path} not found — using built-in defaults.")
    return {}


def build_pose_models(cfg, fall_pose_model_override=None):
    from ultralytics import YOLO
    general = cfg.get("general", {})
    yolo_model = general.get("yolo_model", "yolov8n-pose.pt")
    person_conf = general.get("person_conf", 0.30)
    fall_pose_model = fall_pose_model_override or general.get("fall_pose_model")
    fall_pose_conf = general.get("fall_pose_conf", person_conf)
    class_map = {int(k): v for k, v in general.get("fall_pose_class_map", {0: "laying", 1: "standing"}).items()}

    print(f"[eval] Loading generic pose model: {yolo_model}")
    yolo = YOLO(yolo_model)

    yolo_fall_pose = None
    if fall_pose_model:
        try:
            print(f"[eval] Loading custom fall-pose model: {fall_pose_model}")
            yolo_fall_pose = YOLO(fall_pose_model)
        except Exception as e:
            print(f"⚠️  [eval] Could not load custom fall-pose model ({e}) — using generic model only.")

    return yolo, yolo_fall_pose, person_conf, fall_pose_conf, class_map


def detect_people(frame, yolo, yolo_fall_pose, person_conf, fall_pose_conf, class_map):
    h, w = frame.shape[:2]

    def _parse(results, with_posture):
        out = []
        if not results or results[0].boxes is None:
            return out
        boxes = results[0].boxes
        kpts = getattr(results[0], "keypoints", None)
        try:
            xy = boxes.xyxy.cpu().numpy()
        except Exception:
            xy = np.array(boxes.xyxy)
        try:
            confs = boxes.conf.cpu().numpy().tolist()
        except Exception:
            confs = list(boxes.conf)
        classes = None
        if with_posture:
            try:
                classes = boxes.cls.cpu().numpy().astype(int).tolist()
            except Exception:
                classes = None
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
        for i, box in enumerate(xy.tolist() if hasattr(xy, "tolist") else xy):
            x1, y1, x2, y2 = [float(v) for v in box]
            conf = float(confs[i]) if i < len(confs) else 0.0
            kp = kpts_all[i] if i < len(kpts_all) else []
            person = {"bbox": [x1, y1, x2, y2], "conf": conf, "keypoints": kp}
            if with_posture and classes is not None and i < len(classes):
                person["posture_class"] = class_map.get(classes[i], str(classes[i]))
                person["posture_conf"] = conf
            out.append(person)
        return out

    # [FIX] Generic YOLO as primary detector (best recall),
    # FallGuard overlays posture_class via IoU matching.
    results = yolo(frame, conf=person_conf, classes=[0], verbose=False)
    people = _parse(results, with_posture=False)

    if yolo_fall_pose is not None and people:
        try:
            fp_results = yolo_fall_pose(frame, conf=fall_pose_conf, verbose=False)
            fp_dets = _parse(fp_results, with_posture=True)

            for person in people:
                best_iou, best_fp = 0.0, None
                px1, py1, px2, py2 = person["bbox"]
                for fp in fp_dets:
                    fx1, fy1, fx2, fy2 = fp["bbox"]
                    ix1 = max(px1, fx1); iy1 = max(py1, fy1)
                    ix2 = min(px2, fx2); iy2 = min(py2, fy2)
                    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    area_p = max(1e-6, (px2-px1)*(py2-py1))
                    area_f = max(1e-6, (fx2-fx1)*(fy2-fy1))
                    iou = inter / (area_p + area_f - inter) if (area_p + area_f - inter) > 0 else 0
                    if iou > best_iou:
                        best_iou, best_fp = iou, fp
                if best_fp is not None and best_iou >= 0.3:
                    person["posture_class"] = best_fp.get("posture_class")
                    person["posture_conf"] = best_fp.get("posture_conf", 0.0)
        except Exception as e:
            print(f"[eval] fall-pose overlay error (continuing without posture): {e}")

    return people


def parse_gt(args):
    """Returns dict: {"fall": [...], "hand_sos": [...], "distractor": [...]}"""
    gt = {"fall": [], "hand_sos": [], "distractor": []}
    if args.no_gt:
        return None
    raw_entries = []
    if args.gt_file:
        raw = Path(args.gt_file).read_text()
        raw_entries = [v.strip() for v in raw.replace(",", "\n").splitlines() if v.strip()]
    elif args.gt:
        if args.gt.strip().lower() == "none":
            return gt  # explicitly: zero real events in this video
        raw_entries = [v.strip() for v in args.gt.split(",") if v.strip()]
    else:
        return None

    for entry in raw_entries:
        if ":" in entry:
            ts_str, typ = entry.split(":", 1)
            typ = typ.strip().lower()
        else:
            ts_str, typ = entry, "fall"
        ts = float(ts_str.strip())
        if typ not in gt:
            print(f"⚠️  Unknown ground-truth type '{typ}' in entry '{entry}' — treating as 'fall'.")
            typ = "fall"
        gt[typ].append(ts)
    for k in gt:
        gt[k].sort()
    return gt


def match_events(detected, gt_list, tolerance):
    """detected: list of (ts, tid); gt_list: list of ts. Returns tp, fp, fn, precision, recall, f1."""
    matched_gt, matched_det = set(), set()
    for i, (ts, _tid) in enumerate(detected):
        for j, gt_ts in enumerate(gt_list):
            if j in matched_gt:
                continue
            if abs(ts - gt_ts) <= tolerance:
                matched_gt.add(j); matched_det.add(i)
                break
    tp = len(matched_det)
    fp = len(detected) - tp
    fn = len(gt_list) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2*precision*recall/(precision+recall)) if (recall is not None and (precision+recall) > 0) else None
    return tp, fp, fn, precision, recall, f1


def count_distractor_hits(detected, distractor_list, tolerance):
    hits = 0
    for ts, _tid in detected:
        for d_ts in distractor_list:
            if abs(ts - d_ts) <= tolerance:
                hits += 1
                break
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--gt", default=None, help='e.g. "30:fall,31:hand_sos,70:distractor". Bare numbers default to type "fall". Use --gt none for zero real events.')
    ap.add_argument("--gt-file", default=None)
    ap.add_argument("--no-gt", action="store_true")
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--fall-pose-model", default=None)
    ap.add_argument("--frame-skip", type=int, default=None)
    ap.add_argument("--skip-hand-sos", action="store_true", help="Only evaluate fall (skip hand_sos entirely, e.g. if hand_sos_detector.py isn't available)")
    args = ap.parse_args()

    if args.gt is None and args.gt_file is None and not args.no_gt:
        print("❌ No ground truth given. Pick one:\n"
              '   --gt "30:fall,31:hand_sos,70:distractor"\n'
              '   --gt none            (video has zero real events)\n'
              '   --no-gt              (counts only, no accuracy)')
        sys.exit(1)

    run_hand_sos = HAND_SOS_AVAILABLE and not args.skip_hand_sos
    if not HAND_SOS_AVAILABLE and not args.skip_hand_sos:
        print(f"⚠️  Could not import detectors/hand_sos_detector.py ({_hand_sos_import_error}).\n"
              f"   Make sure that file is present next to this script. Continuing with FALL ONLY.\n"
              f"   (pass --skip-hand-sos to silence this warning)")

    gt = parse_gt(args)

    cfg = load_cfg()
    general = cfg.get("general", {})
    hand_cfg = cfg.get("hand_sos", {})
    frame_skip = args.frame_skip if args.frame_skip is not None else general.get("frame_skip", 1)

    yolo, yolo_fall_pose, person_conf, fall_pose_conf, class_map = build_pose_models(cfg, args.fall_pose_model)
    fall_d = FallDetector(cfg.get("fall", {}))
    hand_d = HandSOSDetector(hand_cfg) if run_hand_sos else None

    # [mirrors main.py] hand_sos per-track state config
    hand_temporal_window = max(1, int(hand_cfg.get("temporal_window", 10)))
    hand_temporal_threshold = min(1.0, max(0.0, float(hand_cfg.get("temporal_threshold", 0.4))))
    hand_temporal_hits_required = max(1, ceil(hand_temporal_window * hand_temporal_threshold))
    hand_min_bbox_area_norm = max(0.0, float(hand_cfg.get("min_hand_bbox_area_norm", 0.0)))
    hand_min_track_age_seconds = max(0.0, float(hand_cfg.get("min_track_age_seconds", 0.8)))
    MIN_CONSEC3 = max(1, int(hand_cfg.get("min_consec_sos_frames", 3)))
    GRACE_FRAMES = 5

    hand_states, hand_miss, hand_first_seen, hand_consec3 = {}, {}, {}, {}
    hand_recent_sos = defaultdict(lambda: deque(maxlen=hand_temporal_window))
    hand_ev = {}
    fall_ev = {}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration_s = (total_frames / fps) if fps else None
    print(f"[eval] Video: {args.video} ({w}x{h}, {fps:.1f}fps"
          f"{f', {duration_s:.1f}s' if duration_s else ''}) | hand_sos={'ON' if run_hand_sos else 'OFF'}")

    tracker = SimpleTracker(frame_w=w, frame_h=h,
                             iou_thresh=general.get("tracker_iou_thresh", 0.3),
                             center_dist_norm=general.get("tracker_center_dist_norm", 0.12))

    detected_fall, detected_hand_sos = [], []
    prev_fallen_tids, prev_sos_confirmed_tids = set(), set()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % max(1, frame_skip) != 0:
            continue
        ts_sec = frame_idx / fps

        frame_clahe = frame
        if run_hand_sos:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            frame_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        people = detect_people(frame, yolo, yolo_fall_pose, person_conf, fall_pose_conf, class_map)
        assigned = tracker.update(people) if people else []

        current_fallen_tids, current_sos_confirmed_tids = set(), set()

        for p in assigned:
            tid = p["track_id"]
            bbox = p["bbox"]
            kp = np.array(p.get("keypoints") or np.zeros((17, 3)))
            posture_class = p.get("posture_class")
            posture_conf = float(p.get("posture_conf", 0.0))

            # ── Fall ──
            fr = fall_d.process(tid, kp, bbox, h, w, posture_class=posture_class, posture_conf=posture_conf)
            if fr["is_fallen"]:
                current_fallen_tids.add(tid)
                if tid not in prev_fallen_tids:
                    detected_fall.append((round(ts_sec, 1), tid))
                    print(f"  [FALL]     @ {ts_sec:.1f}s tid={tid}")
                fall_ev[tid] = True
            else:
                fall_ev[tid] = False

            # ── Hand SOS ── [mirrors main.py per-track state machine + suppression]
            if run_hand_sos:
                prev_hs = hand_states.get(tid, 0)
                hs = prev_hs
                hdet = False
                if tid not in hand_first_seen:
                    hand_first_seen[tid] = ts_sec
                x1, y1, x2, y2 = bbox
                bbox_area_norm = (max(0.0, x2-x1) * max(0.0, y2-y1)) / max(1.0, float(w*h))
                track_age_sec = ts_sec - hand_first_seen[tid]
                hand_eligible = (bbox_area_norm >= hand_min_bbox_area_norm and
                                  track_age_sec >= hand_min_track_age_seconds)
                try:
                    if hand_eligible:
                        hand_lms = hand_d.process_crop(frame_clahe, bbox)
                        if hand_lms:
                            hl = hand_lms[0]
                            hdet = True
                            hs = hand_d.check_sos_step(hs, hl)
                except Exception as e:
                    pass

                if hdet:
                    hand_miss[tid] = 0
                else:
                    hand_miss[tid] = hand_miss.get(tid, 0) + 1
                    if hand_miss[tid] >= GRACE_FRAMES:
                        hs = max(0, hs - 1)
                        hand_miss[tid] = 0
                hand_states[tid] = hs

                if hs == 3 and hdet:
                    hand_consec3[tid] = hand_consec3.get(tid, 0) + 1
                else:
                    hand_consec3[tid] = 0
                consec_ok = hand_consec3.get(tid, 0) >= MIN_CONSEC3
                is_sos_frame = bool(hdet and hs == 3 and hand_eligible and consec_ok)
                recent_sos = hand_recent_sos[tid]
                recent_sos.append(is_sos_frame)
                sos_hits = sum(1 for ok in recent_sos if ok)
                temporal_confirmed = (len(recent_sos) >= hand_temporal_hits_required and
                                       sos_hits >= hand_temporal_hits_required)

                # [PRODUCTION SUPPRESSION RULE — main.py] hand_sos does not
                # fire while a fall event is already active on this track.
                sos_confirmed = temporal_confirmed and not fall_ev.get(tid)

                if sos_confirmed:
                    current_sos_confirmed_tids.add(tid)
                    if tid not in prev_sos_confirmed_tids:
                        detected_hand_sos.append((round(ts_sec, 1), tid))
                        print(f"  [HAND_SOS] @ {ts_sec:.1f}s tid={tid}")

        prev_fallen_tids = current_fallen_tids
        prev_sos_confirmed_tids = current_sos_confirmed_tids

    cap.release()

    print("\n" + "=" * 60)
    print("DETECTED EVENTS")
    print("=" * 60)
    for ts, tid in detected_fall:
        print(f"  FALL      {ts:>7.1f}s   tid={tid}")
    for ts, tid in detected_hand_sos:
        print(f"  HAND_SOS  {ts:>7.1f}s   tid={tid}")
    if not detected_fall and not detected_hand_sos:
        print("  (none)")

    if gt is None:
        print("\n⚠️  NO GROUND TRUTH — counts/timeline only, no precision/recall/accuracy computed.")
        return

    report = {"tolerance_sec": args.tolerance, "total_video_seconds": duration_s, "metrics": {}}

    for label, detected, gt_key in [("fall", detected_fall, "fall"), ("hand_sos", detected_hand_sos, "hand_sos")]:
        tp, fp, fn, precision, recall, f1 = match_events(detected, gt[gt_key], args.tolerance)
        print(f"\n--- {label.upper()} ---")
        print(f"GT: {len(gt[gt_key])}  Detected: {len(detected)}  TP: {tp}  FP: {fp}  FN: {fn}")
        print(f"Precision: {precision:.3f}" if precision is not None else "Precision: N/A")
        print(f"Recall:    {recall:.3f}" if recall is not None else "Recall:    N/A")
        print(f"F1:        {f1:.3f}" if f1 is not None else "F1:        N/A")
        report["metrics"][label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        }

    if gt["distractor"]:
        all_detected = detected_fall + detected_hand_sos
        d_hits = count_distractor_hits(all_detected, gt["distractor"], args.tolerance)
        print(f"\n--- DISTRACTOR SPECIFICITY ---")
        print(f"Distractor poses (should NOT trigger anything): {len(gt['distractor'])}")
        print(f"Detections landing on a distractor: {d_hits}  "
              f"({'⚠️ specificity problem' if d_hits else '✅ clean'})")
        report["distractor"] = {"count": len(gt["distractor"]), "false_triggers": d_hits}

    report["detected_fall"] = detected_fall
    report["detected_hand_sos"] = detected_hand_sos
    report["ground_truth"] = gt

    out_path = Path(args.video).with_suffix("").as_posix() + "_multi_eval_report.json"
    Path(out_path).write_text(json.dumps(report, indent=2))
    print(f"\nJSON report saved: {out_path}")


if __name__ == "__main__":
    main()
