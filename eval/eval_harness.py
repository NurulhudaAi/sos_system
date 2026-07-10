#!/usr/bin/env python3
"""
eval/eval_harness.py — Offline accuracy evaluation for the SOS detection pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

จุดประสงค์
──────────
วัด precision / recall / F1 / false-alarm-per-hour ของ fall / hand_sos / pose_sos /
object_theft / object_left จริงๆ แทนที่จะจูน threshold ด้วยตาอย่างเดียว

หลักการออกแบบ
──────────────
- ไม่แก้ main.py และไม่ import main.py (main.py มี side-effect ตอน import: เรียก
  init_env() ซึ่งอาจ sys.exit(1), ต้องมี .env ที่ configure แล้ว ฯลฯ — ไม่เหมาะกับ
  offline batch script)
- Import detector class ตัวจริง (FallDetector, HandSOSDetector, PoseSOSDetector,
  ObjectGuardian) และ CooldownEngine/ZoneManager/AlertDispatcher จาก pipeline.py
  ตัวเดียวกับที่ production ใช้ — เพื่อไม่ให้ eval กับของจริงเพี้ยนกัน
- ไม่เขียนลง MongoDB เลย (ไม่เรียก insert_incident/insert_object_event) — รันได้
  โดยไม่ต้องมี .env / MongoDB Atlas connection
- ต้องมี model_server.py รันอยู่ก่อน (เหมือน production) เพื่อความ fidelity ของ
  YOLO person/object detection + keypoints

วิธีใช้
───────
    # Terminal 1
    python3 model_server.py

    # Terminal 2
    python3 eval/eval_harness.py --labels eval/labels --videos-dir /path/to/videos

Label schema: ดู eval/labels/example.json
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from detectors.fall_detector import FallDetector          # noqa: E402
from detectors.hand_sos_detector import HandSOSDetector    # noqa: E402
from detectors.pose_sos_detector import PoseSOSDetector    # noqa: E402
from detectors.object_guardian import ObjectGuardian       # noqa: E402
from pipeline import CooldownEngine, ZoneManager, AlertDispatcher  # noqa: E402
from utils import preprocess                               # noqa: E402

# ── Debug toggle ────────────────────────────────────────────────
DEBUG_HAND = True   # ตั้ง False เมื่อ debug เสร็จแล้ว เพื่อลด log ท่วม

def _hand_dbg(msg: str):
    if DEBUG_HAND:
        print(f"  [hand-dbg] {msg}")

# ───────────────────────── Minimal tracking helpers (mirrors main.py) ─────────
# หมายเหตุ: คัดลอกมาจาก main.py โดยตั้งใจ (ไม่ import main.py ตามเหตุผลด้านบน)
# ถ้าแก้ tracking logic ใน main.py ต้องแก้ตรงนี้ด้วย — แนะนำให้ในอนาคตย้าย
# SimpleTracker/BoxesWrapper/KeypointsWrapper ไปเป็น shared module (tracking_utils.py)
# ที่ทั้ง main.py และ eval_harness.py import ร่วมกัน เพื่อไม่ให้ logic 2 ที่หลุดกัน

class _W:
    def __init__(self, v): self.val = v
    def cpu(self): return self
    def numpy(self): return np.array(self.val)
    def __float__(self):
        try: return float(self.val)
        except Exception: return float(np.array(self.val))
    def __int__(self): return int(self.__float__())


class BoxesWrapper:
    def __init__(self, xyxy, confs, ids):
        self.xyxy = [_W(x) for x in xyxy]; self.conf = [_W(c) for c in confs]
        self.id = [_W(i) for i in ids] if ids else None


class KeypointsWrapper:
    def __init__(self, data): self.data = [_W(d) for d in data] if data else None


class SimpleTracker:
    def __init__(self): self.next_id = 0; self.tracks = {}

    def _iou(self, a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        w = max(0, x2 - x1); h = max(0, y2 - y1); inter = w * h
        aa = max(1e-6, (a[2]-a[0])*(a[3]-a[1])); ab = max(1e-6, (b[2]-b[0])*(b[3]-b[1]))
        return inter/(aa+ab-inter) if (aa+ab-inter) > 0 else 0.0

    def update(self, dets):
        used = []
        for det in dets:
            best_id = None; best_iou = 0.0
            for tid, t in self.tracks.items():
                iou = self._iou(det["bbox"], t["bbox"])
                if iou > best_iou: best_iou = iou; best_id = tid
            if best_iou >= 0.3 and best_id not in used:
                det["track_id"] = best_id; self.tracks[best_id].update(bbox=det["bbox"], lost=0); used.append(best_id)
            else:
                tid = self.next_id; self.next_id += 1
                det["track_id"] = tid; self.tracks[tid] = {"bbox": det["bbox"], "lost": 0}; used.append(tid)
        tid_set = {d["track_id"] for d in dets}
        for tid in list(self.tracks):
            if tid not in tid_set:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > 5: del self.tracks[tid]
        return dets

    def cleanup_track(self, tid: int):
        self.tracks.pop(tid, None)


def call_detect_all(det_url: str, frame, timeout=5) -> dict:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    try:
        r = requests.post(det_url + "/detect_all",
                           files={"image": ("f.jpg", buf.tobytes(), "image/jpeg")},
                           timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"[eval] model_server request failed: {e}")
    return {}


def _cooldown_ratio(engine: CooldownEngine) -> Optional[float]:
    """สัดส่วนเฟรม positive ใน buffer ปัจจุบันของ CooldownEngine — ใช้เป็น confidence
    proxy สำหรับ hand_sos/pose_sos (สอดคล้องกับค่าที่ engine ใช้ตัดสินใจ trigger จริง)"""
    buf = engine._buf
    if not buf:
        return None
    return round(sum(buf) / len(buf), 3)


# ───────────────────────── Data classes ────────────────────────────────────

@dataclass
class Prediction:
    video: str
    event_type: str
    track_id: int
    t_sec: float
    confidence: Optional[float]
    matched: bool = False


@dataclass
class GTWindow:
    event_type: str
    start: float
    end: float
    matched: bool = False


# ───────────────────────── Core per-video runner ───────────────────────────

class VideoEvaluator:
    def __init__(self, cfg: dict, det_url: str, zones_path: str):
        self.cfg = cfg
        self.det_url = det_url
        gen = cfg.get("general", {})
        self.W, self.H = 1920, 1080
        self.skip = gen.get("frame_skip", 2)
        self.zones = ZoneManager(zones_path, "default")
        # สร้าง AlertDispatcher ตัวเดียว reuse เฉพาะ _assess_alert_level(atype, extra)
        # (ไม่เรียก .dispatch() เลย → ไม่มี snapshot file / webhook / MongoDB เขียนเกิดขึ้น)
        self._assess = AlertDispatcher(reset_check_interval=999999)._assess_alert_level

    def run(self, video_path: Path, location: str = "eval") -> List[Prediction]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[eval] ❌ cannot open video: {video_path}")
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        tracker = SimpleTracker()
        fall_d = FallDetector(self.cfg.get("fall", {}))
        hand_d = HandSOSDetector(self.cfg.get("hand_sos", {}))
        pose_d = PoseSOSDetector(self.cfg.get("pose_sos", {}))
        obj_grd = ObjectGuardian({**self.cfg.get("object_guardian", {}), "alert_dir": "/tmp/eval_alerts"})

        hand_states: Dict[int, int] = {}
        hand_miss:   Dict[int, int] = {}
        GRACE_FRAMES = 5
        hand_ev: Dict[int, bool] = {}
        hand_last_dispatch: Dict[int, float] = {}   # ← เพิ่มบรรทัดนี้
        HAND_SOS_COOLDOWN_SEC = 3.0                  # ← เพิ่มบรรทัดนี้
        fall_ev: Dict[int, bool] = {}
        pose_cd: Dict[int, CooldownEngine] = {}

        preds: List[Prediction] = []
        source_id = str(video_path)
        n = 0
        while True:
            ret, raw = cap.read()
            if not ret:
                break
            n += 1
            if n % max(1, self.skip) != 0:
                continue
            t_sec = n / fps

            frame = preprocess(raw, self.W, self.H)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # CLAHE brightness enhancement for low-light hand detection
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            rgb_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            try: hand_d.process_frame(rgb_enhanced)
            except Exception: pass

            resp = call_detect_all(self.det_url, frame)
            pdets = resp.get("people", [])
            odets = resp.get("objects", [])

            # ── Object Guardian ──────────────────────────────────────
            gp = [{"bbox": d["bbox"], "track_id": i} for i, d in enumerate(pdets)]
            for oa in obj_grd.update(frame, odets, gp, source_id=source_id, location=location):
                preds.append(Prediction(
                    video=video_path.name,
                    event_type=oa.get("event_type", "object_event"),
                    track_id=oa.get("track_id") or -1,
                    t_sec=t_sec,
                    confidence=float(oa.get("confidence", 0.0)),
                ))

            if not pdets:
                continue
            assigned = tracker.update(pdets)
            boxes = BoxesWrapper([d["bbox"] for d in assigned],
                                  [d.get("conf", 0.0) for d in assigned],
                                  [d.get("track_id") for d in assigned])
            kpts = KeypointsWrapper([d.get("keypoints", []) for d in assigned])

            if boxes.id:
                active = {int(boxes.id[i].cpu()) for i in range(len(boxes.xyxy))}
                dropped = set(hand_states.keys()) - active
                for tid in dropped:
                    tracker.cleanup_track(tid)
                    for d in [hand_states, hand_miss, hand_ev, hand_last_dispatch, fall_ev, pose_cd]:
                        d.pop(tid, None)

            if not kpts.data:
                continue
            for i in range(len(boxes.xyxy)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu())
                tid = int(boxes.id[i].cpu()) if boxes.id else i
                if i >= len(kpts.data): continue
                kp = kpts.data[i].cpu().numpy()
                x1, y1, x2, y2 = bbox
                if not self.zones.in_zone((x1+x2)/2/w, (y1+y2)/2/h): continue

                fr = fall_d.process(tid, kp, bbox, h, w)

                # ── Hand SOS (state machine, เหมือน main.py) ─────────
                hs = hand_states.get(tid, 0)
                hdet = False
                try:
                    rh = getattr(hand_d, "_results", None)
                    if rh and getattr(rh, "hand_landmarks", None):
                        for hidx, hl in enumerate(rh.hand_landmarks):
                            xs = [l.x for l in hl]; ys = [l.y for l in hl]
                            area = (max(xs)-min(xs))*w*(max(ys)-min(ys))*h/(w*h)
                            min_area = self.cfg.get("hand_sos", {}).get("min_hand_bbox_area_norm", 0.002)
                            if area < min_area:
                                _hand_dbg(f"t={t_sec:.2f}s tid={tid} hand#{hidx} "
                                          f"area={area:.5f} min={min_area:.5f} SKIP(small)")
                                continue
                            pts = [(int(l.x*w), int(l.y*h)) for l in hl]
                            hx = sum(p[0] for p in pts)/len(pts); hy = sum(p[1] for p in pts)/len(pts)
                            in_bbox = x1 <= hx <= x2 and y1 <= hy <= y2
                            _hand_dbg(f"t={t_sec:.2f}s tid={tid} hand#{hidx} "
                                      f"area={area:.5f} min={min_area:.5f} OK")
                            _hand_dbg(f"  hand_center=({int(hx)},{int(hy)}) "
                                      f"person_bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)}) in_bbox={in_bbox}")
                            if in_bbox:
                                hdet = True
                                try:
                                    p_open  = hand_d._palm_open(hl)
                                    t_in    = hand_d._thumb_in(hl)
                                    f_close = hand_d._fingers_closed(hl)
                                    _hand_dbg(f"  state={hs} palm_open={p_open} "
                                              f"thumb_in={t_in} fingers_closed={f_close}")
                                    if hs == 0 and p_open: hs = 1
                                    elif hs == 1 and t_in: hs = 2
                                    elif hs == 2 and f_close: hs = 3
                                except Exception as e:
                                    _hand_dbg(f"  !! helper method error: {e}")
                                break
                except Exception as e:
                    _hand_dbg(f"t={t_sec:.2f}s tid={tid} !! detection block error: {e}")

                prev_hs = hand_states.get(tid, 0)
                if hdet:
                    hand_miss[tid] = 0
                else:
                    hand_miss[tid] = hand_miss.get(tid, 0) + 1
                    if hand_miss[tid] >= GRACE_FRAMES:
                        hs = max(0, hs - 1)
                        hand_miss[tid] = 0
                hand_states[tid] = hs
                if hs != prev_hs or hdet:
                    _hand_dbg(f"t={t_sec:.2f}s tid={tid} state: {prev_hs}→{hs} hdet={hdet}")

                if hs == 3 and not fall_ev.get(tid):
                    hand_ev[tid] = True
                    _hand_dbg(f"t={t_sec:.2f}s tid={tid} *** STATE 3 REACHED — hand_ev=True ***")
                elif hand_ev.get(tid):
                    last = hand_last_dispatch.get(tid, -999)
                    if t_sec - last >= HAND_SOS_COOLDOWN_SEC:
                        preds.append(Prediction(video=video_path.name, event_type="hand_sos",
                                                 track_id=tid, t_sec=t_sec, confidence=1.0))
                        hand_last_dispatch[tid] = t_sec
                        _hand_dbg(f"t={t_sec:.2f}s tid={tid} >>> falling-edge dispatch — prediction recorded")
                    else:
                        _hand_dbg(f"t={t_sec:.2f}s tid={tid} >>> falling-edge suppressed "
                                  f"(cooldown {t_sec-last:.2f}s < {HAND_SOS_COOLDOWN_SEC}s)")
                    hand_ev[tid] = False

                # ── Pose SOS ──────────────────────────────────────────
                try:
                    pose_r = pose_d.detect(kp, h)
                except Exception:
                    pose_r = {"is_sos": False}
                pcd = pose_cd.get(tid)
                if pcd is None:
                    pcd = CooldownEngine("pose_sos", self.cfg.get("pose_sos", {}))
                    pose_cd[tid] = pcd
                if pcd.update(bool(pose_r.get("is_sos"))):
                    preds.append(Prediction(video=video_path.name, event_type="pose_sos",
                                             track_id=tid, t_sec=t_sec,
                                             confidence=_cooldown_ratio(pcd)))

                # ── Fall ──────────────────────────────────────────────
                esc = fr.get("danger_lying") and not fr.get("recovered_quickly")
                is_critical = fr.get("is_critical") and not fr.get("recovered_quickly")
                is_confirmed = (fr.get("is_fallen") or esc) and not fr.get("recovered_quickly")
                if is_critical or is_confirmed:
                    if not fall_ev.get(tid):
                        ex = {"track_id": tid, "source": source_id, "location": location,
                              "recovered_quickly": fr.get("recovered_quickly"), "fall_result": fr}
                        if is_critical: ex["critical"] = True
                        if esc: ex["auto_escalated_immobile"] = True
                        lv, ln, _flags = self._assess("fall", ex)
                        preds.append(Prediction(video=video_path.name, event_type="fall",
                                                 track_id=tid, t_sec=t_sec,
                                                 confidence=round((lv+1)/4, 3)))
                        fall_ev[tid] = True
                else:
                    fall_ev[tid] = False

        # ── สิ้นสุดวิดีโอ: flush event ที่ยังค้างอยู่ (state=3 แต่ไม่เคย falling-edge จะ trigger) ──
        for tid, active in list(hand_ev.items()):
            if active:
                last = hand_last_dispatch.get(tid, -999)
                if t_sec - last >= HAND_SOS_COOLDOWN_SEC:
                    preds.append(Prediction(video=video_path.name, event_type="hand_sos",
                                             track_id=tid, t_sec=t_sec, confidence=1.0))
                    hand_last_dispatch[tid] = t_sec
                    _hand_dbg(f"t={t_sec:.2f}s tid={tid} >>> END-OF-VIDEO FLUSH — "
                              f"event ค้างที่ state=3 ตอนวิดีโอจบ, บันทึกเป็น prediction")
                hand_ev[tid] = False
        cap.release()
        try: hand_d.release()
        except Exception: pass
        return preds


# ───────────────────────── Matching / metrics ──────────────────────────────

def match_predictions(preds: List[Prediction], gt_windows: List[GTWindow], tolerance: float):
    """สำหรับ predictions/gt ของวิดีโอเดียว, event_type เดียวกัน (เรียกทีละคู่)"""
    windows = [GTWindow(g.event_type, g.start - tolerance, g.end + tolerance) for g in gt_windows]
    tp = fp = 0
    for p in sorted(preds, key=lambda x: x.t_sec):
        matched_any = False
        matched_new = False
        for wgt, w in zip(gt_windows, windows):
            if w.start <= p.t_sec <= w.end:
                matched_any = True
                if not wgt.matched:
                    wgt.matched = True
                    matched_new = True
                break
        if matched_new:
            tp += 1
            p.matched = True
        elif not matched_any:
            fp += 1
    fn = sum(1 for g in gt_windows if not g.matched)
    return tp, fp, fn


def evaluate(all_preds: Dict[str, List[Prediction]],
             all_gt: Dict[str, List[GTWindow]],
             tolerance: float):
    event_types = set()
    for preds in all_preds.values():
        event_types.update(p.event_type for p in preds)
    for gts in all_gt.values():
        event_types.update(g.event_type for g in gts)

    report = {}
    for et in sorted(event_types):
        tp_total = fp_total = fn_total = 0
        for video in set(list(all_preds.keys()) + list(all_gt.keys())):
            preds = [p for p in all_preds.get(video, []) if p.event_type == et]
            gts = [g for g in all_gt.get(video, []) if g.event_type == et]
            tp, fp, fn = match_predictions(preds, gts, tolerance)
            tp_total += tp; fp_total += fp; fn_total += fn
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else None
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else None
        f1 = (2*precision*recall/(precision+recall)
              if precision is not None and recall is not None and (precision+recall) > 0 else None)
        report[et] = {"tp": tp_total, "fp": fp_total, "fn": fn_total,
                       "precision": precision, "recall": recall, "f1": f1}
    return report


# ───────────────────────── CLI / main ───────────────────────────────────────

def load_labels(labels_dir: Path) -> Dict[str, List[GTWindow]]:
    all_gt: Dict[str, List[GTWindow]] = defaultdict(list)
    for f in sorted(labels_dir.glob("*.json")):
        if f.name == "example.json":
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        video = d["video"]
        for ev in d.get("events", []):
            all_gt[video].append(GTWindow(ev["event_type"], float(ev["start_sec"]), float(ev["end_sec"])))
    return all_gt


def video_duration_hours(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return (frames / fps) / 3600.0 if fps else 0.0


def main():
    ap = argparse.ArgumentParser(description="Offline evaluation harness for SOS detection pipeline")
    ap.add_argument("--labels", default="eval/labels", help="โฟลเดอร์ label JSON (default: eval/labels)")
    ap.add_argument("--videos-dir", required=True, help="โฟลเดอร์ที่เก็บไฟล์วิดีโอ (ชื่อไฟล์ต้องตรงกับ 'video' ใน label)")
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "thresholds.yaml"))
    ap.add_argument("--zones", default=str(REPO_ROOT / "config" / "zones.yaml"))
    ap.add_argument("--detector-url", default="http://127.0.0.1:8000")
    ap.add_argument("--tolerance-sec", type=float, default=5.0,
                    help="ยอมให้ detection ล่าช้า/เร็วกว่า label กี่วินาที (default 5.0)")
    ap.add_argument("--out", default="eval/report")
    args = ap.parse_args()

    # health check model_server ก่อนเริ่ม
    try:
        r = requests.get(args.detector_url + "/health", timeout=3)
        if r.status_code != 200:
            raise RuntimeError(f"status {r.status_code}")
    except Exception as e:
        print(f"❌ model_server ไม่ตอบสนองที่ {args.detector_url} ({e})")
        print("   รัน `python3 model_server.py` ในอีก terminal ก่อน แล้วค่อยรันสคริปต์นี้ใหม่")
        sys.exit(1)

    cfg = yaml.safe_load(Path(args.config).read_text())
    labels_dir = Path(args.labels)
    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    all_gt = load_labels(labels_dir)
    evaluator = VideoEvaluator(cfg, args.detector_url, args.zones)

    all_preds: Dict[str, List[Prediction]] = {}
    total_hours = 0.0
    for video_name in all_gt.keys():
        video_path = videos_dir / video_name
        if not video_path.exists():
            print(f"⚠️  ไม่พบไฟล์วิดีโอ: {video_path} (ข้าม)")
            continue
        print(f"▶ กำลังประมวลผล: {video_name} ...")
        t0 = time.time()
        preds = evaluator.run(video_path)
        all_preds[video_name] = preds
        total_hours += video_duration_hours(video_path)
        print(f"  เสร็จใน {time.time()-t0:.1f}s — พบ {len(preds)} predicted events")

    report = evaluate(all_preds, all_gt, args.tolerance_sec)

    total_fp = sum(v["fp"] for v in report.values())
    fa_per_hour = round(total_fp / total_hours, 3) if total_hours > 0 else None

    print("\n" + "="*70)
    print(f"{'Event Type':<16}{'TP':>5}{'FP':>5}{'FN':>5}{'Precision':>12}{'Recall':>10}{'F1':>8}")
    print("-"*70)
    for et, m in report.items():
        p = f"{m['precision']:.2f}" if m['precision'] is not None else "  -"
        r = f"{m['recall']:.2f}" if m['recall'] is not None else "  -"
        f1 = f"{m['f1']:.2f}" if m['f1'] is not None else "  -"
        print(f"{et:<16}{m['tp']:>5}{m['fp']:>5}{m['fn']:>5}{p:>12}{r:>10}{f1:>8}")
    print("-"*70)
    print(f"Total video time: {total_hours:.2f} hr | False alarms/hour (all types): {fa_per_hour}")
    print("="*70 + "\n")

    # เขียน report ละเอียด
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps({
        "tolerance_sec": args.tolerance_sec,
        "total_video_hours": total_hours,
        "false_alarms_per_hour": fa_per_hour,
        "metrics": report,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / "raw_predictions.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("video,event_type,track_id,t_sec,confidence,matched\n")
        for video, preds in all_preds.items():
            for p in preds:
                f.write(f"{video},{p.event_type},{p.track_id},{p.t_sec:.2f},{p.confidence},{p.matched}\n")

    print(f"- บันทึกผลละเอียดที่: {report_path}")
    print(f"- บันทึก raw predictions ที่: {csv_path}")


if __name__ == "__main__":
    main()
