#!/usr/bin/env python3
"""
eval/live_full_rtsp.py

Live monitor แบบเต็ม จาก RTSP stream — รวมทั้ง 3 detector:
    - FallDetector      → insert_incident(event_type="fall")
    - HandSOSDetector   → insert_incident(event_type="hand_sos")
    - ObjectGuardian    → insert_object_event(event_type="left_behind"|"theft")

Flow (ตาม references/detection-flow.md):
    cap.read() → frame
        → POST /detect_all → {"people": [...], "objects": [...]}
        → ObjectGuardian.update(...)                 → insert_object_event()
        → per person: FallDetector.process(...)      → insert_incident("fall")
        → per person: HandSOSDetector.process(...)   → insert_incident("hand_sos")

⚠️ หมายเหตุ:
    - HandSOSDetector.process_frame(frame_rgb) รันครั้งเดียวต่อเฟรม (ไม่ใช่ต่อคน)
      ผลเก็บใน hand_d._results — สคริปต์นี้จับคู่มือกับคนด้วย in_bbox check (wrist
      point อยู่ใน bbox คนไหน) แล้วรัน state machine 0→1→2→3 เองที่นี่ ตาม design จริง
    - field ชื่อ keypoints ใน response /detect_all ยังไม่ยืนยัน 100% — ลองหลายชื่อ
      ที่พบบ่อยไว้ให้แล้ว (ดู _KP_KEYS) ถ้าเจอ warning ให้เช็ค response จริงแล้วเพิ่ม
    - AlertDispatcher import path ยังไม่ยืนยัน — ถ้าหาไม่เจอจะ fallback เป็น dispatcher
      ง่าย ๆ ในไฟล์นี้เอง (severity/cooldown พื้นฐาน ไม่ตรงกับของจริง)

วิธีรัน:
    python3 eval/live_full_rtsp.py \
        --rtsp-url "rtsp://mfustream:mediamfu2025@172.28.106.79/Streaming/Channels/101" \
        --detector-url http://127.0.0.1:8000 \
        --location "ห้องสมุด" \
        --source-id "cam-101" \
        --sample-fps 5

กด Ctrl+C เพื่อหยุด
"""
import argparse
import os
import signal
import sys
import time
import uuid
from collections import defaultdict, deque
from math import ceil
from pathlib import Path

import cv2
import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── โหลด .env ก่อน import database.py เสมอ ──────────────────────────────────
# database.py อ่าน MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
# ที่ module level ตอน import — ถ้า .env ยังไม่ถูกโหลดเข้า os.environ ก่อนบรรทัดนี้
# มันจะ fallback ไป localhost:27017 เงียบ ๆ (นี่คือสาเหตุจริงของปัญหา connection refused)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    _env_path = _PROJECT_ROOT / ".env"
    loaded = load_dotenv(_env_path)
    if not loaded or not os.environ.get("MONGODB_URI"):
        print(f"⚠️  ไม่พบ MONGODB_URI หลังโหลด {_env_path} — เช็คว่าไฟล์ .env อยู่ที่ project root จริงไหม")
except ImportError:
    print("⚠️  ไม่มี python-dotenv ติดตั้ง — pip install python-dotenv")

from detectors.object_guardian import ObjectGuardian
from detectors.fall_detector import FallDetector
from detectors.hand_sos_detector import HandSOSDetector
from database import insert_incident, health_check  # ต้อง import หลังโหลด .env เท่านั้น

# AlertDispatcher: ลองทุก module name ที่พบได้บ่อยในโปรเจกต์แบบนี้ ก่อน fallback
AlertDispatcher = None
_DISPATCHER_CANDIDATES = ["dispatch", "alert_dispatcher", "dispatcher", "alerts"]
for _mod_name in _DISPATCHER_CANDIDATES:
    try:
        _mod = __import__(_mod_name, fromlist=["AlertDispatcher"])
        AlertDispatcher = getattr(_mod, "AlertDispatcher")
        break
    except (ImportError, AttributeError):
        continue


class _FallbackDispatcher:
    """ใช้ถ้าหา AlertDispatcher ตัวจริงไม่เจอ — ทำแค่ cooldown + snapshot save พื้นฐาน
    ไม่มี severity assessment แบบ project จริง (severity ต้องกำหนดเอง)"""
    def __init__(self, snapshot_dir="logs/snapshots", cooldown_seconds=60):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown_seconds = cooldown_seconds
        self._last_sent = {}

    def dispatch(self, event_type, frame, extra):
        key = (event_type, extra.get("track_id"))
        now = time.time()
        last = self._last_sent.get(key, 0)
        if now - last < self.cooldown_seconds:
            return None
        self._last_sent[key] = now
        fname = f"{event_type}_{extra.get('track_id')}_{int(now)}.jpg"
        path = self.snapshot_dir / fname
        cv2.imwrite(str(path), frame)
        return str(path)

    def _assess_alert_level(self, event_type, extra):
        if extra.get("critical"):
            return 3, "CRITICAL", ["critical"]
        return 2, "HIGH", []


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
        print(f"[detect_all] HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[detect_all] error: {e}")
    return {"people": [], "objects": []}


def _match_hand_to_bbox(hand_landmarks_list, bbox, w, h):
    """หา hand landmarks (list ของ 21 จุด, normalized 0-1) ที่ wrist (lm[0])
    ตกอยู่ใน bbox ของคนนี้ — คืน landmark list แรกที่ match หรือ None ถ้าไม่มี"""
    x1, y1, x2, y2 = bbox
    for lm in hand_landmarks_list:
        wrist = lm[0]
        px, py = wrist.x * w, wrist.y * h
        if x1 <= px <= x2 and y1 <= py <= y2:
            return lm
    return None


def _update_hand_state(hand_d, cur_state, lm):
    """State machine ตาม design จริงของ HandSOSDetector (0→1→2→3):
      0 idle → palm_open() → 1
      1 palm open → thumb_in() → 2 ; ไม่มีมือ → decay กลับ 0
      2 thumb tucked → fingers_closed() → 3 (SOS) ; ไม่มีมือ → decay กลับ 0
      3 SOS confirmed → caller reset กลับ 0 หลัง dispatch
    HandSOSDetector ไม่มี state machine ในตัวเอง (ดู docstring ของคลาส) —
    ต้องจัดการ per-track เองใน caller นี้แหละ ตรงกับที่ comment ในซอร์สบอกไว้
    """
    if lm is None:
        return 0 if cur_state in (1, 2) else cur_state
    if cur_state == 0:
        return 1 if hand_d._palm_open(lm) else 0
    if cur_state == 1:
        return 2 if hand_d._thumb_in(lm) else 1
    if cur_state == 2:
        return 3 if hand_d._fingers_closed(lm) else 2
    return cur_state


def _prd_severity(level_name: str) -> str:
    """แปลง severity ที่ AlertDispatcher คืนมา (LOW/HIGH/CRITICAL แบบเดิม) ให้เป็น
    PRD schema string ที่ database.py คาดหวังจริง: 'High' | 'Medium' | 'Low'"""
    mapping = {
        "CRITICAL": "High",
        "HIGH": "High",
        "MEDIUM": "Medium",
        "LOW": "Low",
    }
    return mapping.get(str(level_name).upper(), "Medium")


def _iso_now() -> str:
    from datetime import datetime, UTC
    return datetime.now(UTC).isoformat()


def open_rtsp(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def run_live(rtsp_url, detector_url, obj_cfg, fall_cfg, hand_cfg,
             location, source_id, sample_fps=5, reconnect_wait_sec=3.0):

    if not health_check():
        print("⚠️  MongoDB health_check() ไม่ผ่าน — เช็ค .env / MONGODB_URI ก่อนรัน")

    guardian = ObjectGuardian({**obj_cfg, "alert_dir": "eval/alerts_object_live"})
    fall_d = FallDetector(fall_cfg)
    hand_d = HandSOSDetector(hand_cfg)
    tracker = SimpleTracker()

    disp = None
    if AlertDispatcher is not None:
        try:
            disp = AlertDispatcher()
        except Exception as e:
            print(f"⚠️  AlertDispatcher() constructor error: {e} — ใช้ fallback dispatcher แทน")
    if disp is None:
        disp = _FallbackDispatcher()
        if AlertDispatcher is None:
            print("⚠️  หา AlertDispatcher ตัวจริงไม่เจอ (import path) — ใช้ fallback dispatcher "
                  "(severity/cooldown แบบง่าย) แก้ import ด้านบนให้ตรงกับ main.py จริง")

    fall_ev = {}     # track_id -> True เมื่อ dispatch ไปแล้ว
    hand_ev = {}     # track_id -> True ระหว่างที่ gesture ยังถูกยืนยัน
    hand_state = {}  # track_id -> state (0-3)
    hand_miss = {}   # [RTSP FIX] track_id -> miss count for grace period
    hand_first_seen = {}
    hand_temporal_window = max(1, int(hand_cfg.get("temporal_window", 10)))
    hand_temporal_threshold = min(1.0, max(0.0, float(hand_cfg.get("temporal_threshold", 0.4))))
    hand_temporal_hits_required = max(1, ceil(hand_temporal_window * hand_temporal_threshold))
    hand_min_bbox_area_norm = max(0.0, float(hand_cfg.get("min_hand_bbox_area_norm", 0.0)))
    hand_min_track_age_seconds = max(0.0, float(hand_cfg.get("min_track_age_seconds", 0.8)))
    hand_cooldown_seconds = max(0.0, float(hand_cfg.get("cooldown_seconds", 60)))
    hand_recent_sos = defaultdict(lambda: deque(maxlen=hand_temporal_window))
    hand_last_event_t = -1e9
    GRACE_FRAMES = 5

    stop = {"flag": False}

    def _handle_sigint(sig, frame):
        stop["flag"] = True
        print("\n⏹  ได้รับสัญญาณหยุด กำลังปิดการเชื่อมต่อ...")

    signal.signal(signal.SIGINT, _handle_sigint)

    print(f"▶ เชื่อมต่อ RTSP: {rtsp_url}")
    print(f"  detector: {detector_url}  |  location: {location}  |  source_id: {source_id}")
    print(f"  sample_fps: {sample_fps}")
    print(f"  ตรวจจับ: fall + hand_sos + object_left/theft")
    print(f"  [RTSP FIX] ใช้ per-person crop + check_sos_step()")
    print("  กด Ctrl+C เพื่อหยุด\n")

    cap = open_rtsp(rtsp_url)
    last_snapshot_t = 0.0
    t_start = time.time()
    frames_seen = 0
    events_seen = {"fall": 0, "hand_sos": 0, "object": 0}

    while not stop["flag"]:
        if not cap.isOpened():
            print(f"⚠️  RTSP ไม่เปิด / หลุด — ลองใหม่ใน {reconnect_wait_sec}s ...")
            cap.release()
            time.sleep(reconnect_wait_sec)
            cap = open_rtsp(rtsp_url)
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"⚠️  อ่านเฟรมไม่ได้ (stream หลุด?) — ลองใหม่ใน {reconnect_wait_sec}s ...")
            cap.release()
            time.sleep(reconnect_wait_sec)
            cap = open_rtsp(rtsp_url)
            continue

        frames_seen += 1
        now = time.time()
        if now - last_snapshot_t < (1.0 / sample_fps):
            continue
        last_snapshot_t = now
        t_sec = now - t_start
        h, w = frame.shape[:2]

        # [RTSP FIX] CLAHE preprocessing for hand detection
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            frame_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception:
            frame_clahe = frame

        resp = detect_all(detector_url, frame)
        people_raw = resp.get("people", [])
        objects_raw = resp.get("objects", [])

        assigned = tracker.update([{"bbox": p["bbox"]} for p in people_raw])
        alive = set(tracker.tracks.keys())
        dropped = set(hand_state.keys()) - alive
        for tid in dropped:
            for d in (hand_state, hand_miss, hand_first_seen, hand_ev, fall_ev):
                d.pop(tid, None)
            hand_recent_sos.pop(tid, None)
        _KP_KEYS = ("keypoints", "kpts", "pose", "landmarks", "kp")
        for det, raw in zip(assigned, people_raw):
            kp = None
            for k in _KP_KEYS:
                if raw.get(k) is not None:
                    kp = raw[k]
                    break
            det["keypoints"] = kp
        if people_raw and all(d.get("keypoints") is None for d in assigned):
            if not getattr(run_live, "_kp_warned", False):
                print("⚠️  /detect_all response ไม่มี field keypoints ที่รู้จัก "
                      f"(ลองแล้ว: {_KP_KEYS}) — FallDetector จะทำงานไม่ได้ผล")
                run_live._kp_warned = True

        gp = [{"bbox": d["bbox"], "track_id": d["track_id"]} for d in assigned]

        # ── ObjectGuardian ──
        odets = [
            {
                "bbox": o["bbox"],
                "confidence": o.get("conf", o.get("confidence", 0.0)),
                "class_name": o.get("class_name", str(o.get("class_id", "object"))),
            }
            for o in objects_raw
        ]
        try:
            guardian_alerts = guardian.update(frame, odets, gp, source_id=source_id,
                                               location=location, t_sec=t_sec)
        except Exception as e:
            print(f"⚠️  ObjectGuardian.update() error: {e}")
            guardian_alerts = []

        for alert in guardian_alerts:
            events_seen["object"] += 1
            print(f"🚨 [{time.strftime('%H:%M:%S')}] object.{alert['event_type']} "
                  f"class={alert.get('class_name')} conf={alert.get('confidence', 0):.2f} "
                  f"owner={alert.get('owner_track_id')} suspect={alert.get('suspect_track_id')}")

        # ── Fall + Hand SOS ต่อคน ──
        for det in assigned:
            tid = det["track_id"]
            bbox = det["bbox"]
            kp = det.get("keypoints")

            # Fall
            fr = None
            try:
                fr = fall_d.process(tid, kp, bbox, h, w)
            except Exception as e:
                if not getattr(run_live, "_fall_err_warned", False):
                    print(f"⚠️  FallDetector.process() error: {e}")
                    run_live._fall_err_warned = True

            if fr is not None:
                is_critical = fr.get("is_critical") and not fr.get("recovered_quickly")
                is_confirmed = (fr.get("is_fallen") or fr.get("danger_lying")) and not fr.get("recovered_quickly")
                if (is_critical or is_confirmed) and not fall_ev.get(tid):
                    ex = {"track_id": tid, "source": source_id, "location": location,
                          "critical": bool(is_critical), "fall_result": fr}
                    img_path = disp.dispatch("fall", frame, ex)
                    if img_path:
                        level, level_name, flags = disp._assess_alert_level("fall", ex)
                        insert_incident(
                            event_uuid=str(uuid.uuid4()),
                            detection_type="Fall",
                            zone=location,
                            confidence=float(fr.get("confidence", 0.9)),
                            timestamp=_iso_now(),
                            severity=_prd_severity(level_name),
                            metadata={
                                "track_id": tid, "source_id": source_id,
                                "image_path": img_path, "flags": flags,
                                "critical": bool(is_critical), "fall_result": fr,
                            },
                        )
                        events_seen["fall"] += 1
                        print(f"🚨 [{time.strftime('%H:%M:%S')}] fall track={tid} severity={level_name}")
                        fall_ev[tid] = True
                elif fr.get("recovered_quickly"):
                    fall_ev[tid] = False

            # [RTSP FIX] Hand SOS — per-person crop + check_sos_step()
            prev_hs = hand_state.get(tid, 0)
            hs = prev_hs
            hdet = False
            if tid not in hand_first_seen:
                hand_first_seen[tid] = time.time()
            x1, y1, x2, y2 = bbox
            bbox_area_norm = (max(0.0, x2 - x1) * max(0.0, y2 - y1)) / max(1.0, float(w * h))
            track_age_sec = now - hand_first_seen[tid]
            hand_eligible = (
                bbox_area_norm >= hand_min_bbox_area_norm and
                track_age_sec >= hand_min_track_age_seconds
            )
            try:
                if hand_eligible:
                    hand_lms = hand_d.process_crop(frame_clahe, bbox)
                    if hand_lms:
                        hl = hand_lms[0]
                        hdet = True
                        hs = hand_d.check_sos_step(hs, hl)
            except Exception:
                pass

            if hdet:
                hand_miss[tid] = 0
            else:
                hand_miss[tid] = hand_miss.get(tid, 0) + 1
                if hand_miss[tid] >= GRACE_FRAMES:
                    hs = max(0, hs - 1)
                    hand_miss[tid] = 0

            hand_state[tid] = hs
            is_sos_frame = bool(hdet and hs == 3 and hand_eligible)
            recent_sos = hand_recent_sos[tid]
            recent_sos.append(is_sos_frame)
            sos_hits = sum(1 for ok in recent_sos if ok)
            temporal_confirmed = len(recent_sos) >= hand_temporal_hits_required and sos_hits >= hand_temporal_hits_required
            fast_sos_confirmed = is_sos_frame and prev_hs >= 1
            sos_confirmed = temporal_confirmed or fast_sos_confirmed

            if sos_confirmed and not hand_ev.get(tid) and not fall_ev.get(tid):
                if t_sec - hand_last_event_t < hand_cooldown_seconds:
                    hand_ev[tid] = True
                    continue
                ex = {"track_id": tid, "source": source_id, "location": location}
                img_path = disp.dispatch("hand_sos", frame, ex)
                if img_path:
                    level, level_name, flags = disp._assess_alert_level("hand_sos", ex)
                    insert_incident(
                        event_uuid=str(uuid.uuid4()),
                        detection_type="Gesture",
                        zone=location,
                        confidence=0.9,
                        timestamp=_iso_now(),
                        severity=_prd_severity(level_name),
                        metadata={
                            "track_id": tid, "source_id": source_id,
                            "image_path": img_path, "flags": flags,
                        },
                    )
                    events_seen["hand_sos"] += 1
                    print(f"🚨 [{time.strftime('%H:%M:%S')}] hand_sos track={tid} severity={level_name}")
                    hand_last_event_t = t_sec
                hand_ev[tid] = True
            elif not sos_confirmed:
                hand_ev[tid] = False

        if frames_seen % (sample_fps * 30) == 0:
            print(f"  ... ยังทำงานอยู่ | frames={frames_seen} | "
                  f"fall={events_seen['fall']} hand_sos={events_seen['hand_sos']} "
                  f"object={events_seen['object']} | uptime={now - t_start:.0f}s")

    cap.release()
    print(f"\n✅ หยุดแล้ว — fall={events_seen['fall']} hand_sos={events_seen['hand_sos']} "
          f"object={events_seen['object']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtsp-url", default="rtsp://mfustream:mediamfu2025@172.28.106.79/Streaming/Channels/101")
    ap.add_argument("--detector-url", default="http://127.0.0.1:8000")
    ap.add_argument("--location", default="")
    ap.add_argument("--source-id", default="cam-101")
    ap.add_argument("--sample-fps", type=int, default=5)
    ap.add_argument("--reconnect-wait-sec", type=float, default=3.0)
    args = ap.parse_args()

    try:
        thresholds = yaml.safe_load(Path("config/thresholds.yaml").read_text())
    except Exception:
        thresholds = {}

    run_live(
        rtsp_url=args.rtsp_url,
        detector_url=args.detector_url,
        obj_cfg=thresholds.get("object_guardian", {}),
        fall_cfg=thresholds.get("fall", {}),
        hand_cfg=thresholds.get("hand_sos", {}),
        location=args.location,
        source_id=args.source_id,
        sample_fps=args.sample_fps,
        reconnect_wait_sec=args.reconnect_wait_sec,
    )


if __name__ == "__main__":
    main()
