#!/usr/bin/env python3
"""
detectors/object_guardian.py
ตรวจจับของลืมทิ้ง (object left-behind) และของถูกขโมย (object theft)

[FIX — tracking key instability]
เดิม: key = f"{class_name}_{int(bbox[0])}_{int(bbox[1])}"
ปัญหา: bbox jitter เล็กน้อยระหว่างเฟรม (แสง/model noise) ทำให้
int(bbox[0]) เปลี่ยนค่าไปมา แม้วัตถุจะอยู่นิ่งสนิท -> key เปลี่ยน ->
ถูกมองว่าเป็น object ใหม่ทุกครั้ง -> first_seen ถูก reset ตลอด ->
elapsed แทบไม่มีทางไต่ถึง left_behind_seconds/theft_seconds ได้เลย
(อาการเดียวกับ state-machine fragility ที่เจอตอน hand_sos ก่อนหน้า)

แก้: จับคู่ track เดิมด้วย IoU (เหมือน SimpleTracker ใน main.py)
แทนตำแหน่งที่ปัดเศษ -> object นิ่งบนโต๊ะยังคง track เดิมต่อเนื่อง
แม้ bbox จะสั่นไปมาบ้างในแต่ละเฟรม
"""
import cv2
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ObjectGuardian:
    def __init__(self, cfg: dict):
        self.cfg                = cfg
        self.alert_dir          = Path(cfg.get("alert_dir", "alerts"))
        self.alert_dir.mkdir(parents=True, exist_ok=True)

        # thresholds
        self.left_seconds       = cfg.get("left_behind_seconds", 30)
        self.theft_seconds      = cfg.get("theft_seconds", 10)
        self.min_confidence     = cfg.get("min_confidence", 0.4)
        self.iou_threshold      = cfg.get("iou_threshold", 0.4)
        # [FIX] separate, lower IoU threshold just for re-identifying the
        # same physical object across frames (person-proximity IoU above
        # stays as-is; this one governs object-to-track matching)
        self.track_iou_threshold = cfg.get("track_iou_threshold", 0.3)
        # [FIX] how many consecutive missed frames before a track is
        # dropped — small tolerance for a missed detection or brief
        # occlusion, so a single dropped frame doesn't wipe first_seen
        self.track_max_missed   = cfg.get("track_max_missed", 5)

        # state
        self._tracked: Dict[int, dict] = {}   # track_id -> state
        self._next_track_id = 0
        self._alerts_sent: set          = set()

    # ─── IOU helper ──────────────────────────────────────────────────────────

    def _iou(self, a, b) -> float:
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        w  = max(0, x2 - x1); h  = max(0, y2 - y1)
        inter = w * h
        aa = max(1e-6, (a[2]-a[0])*(a[3]-a[1]))
        ab = max(1e-6, (b[2]-b[0])*(b[3]-b[1]))
        return inter / (aa + ab - inter)

    def _person_nearby(self, bbox, people: list) -> bool:
        for p in people:
            if self._iou(bbox, p["bbox"]) > self.iou_threshold:
                return True
        return False

    # [FIX] IoU-based track matching — replaces the old rounded-position key
    def _match_track(self, class_name: str, bbox) -> Optional[int]:
        """Find an existing track of the same class whose last-seen bbox
        overlaps this detection above track_iou_threshold. Returns the
        track_id, or None if no match (caller should create a new track)."""
        best_id = None
        best_iou = 0.0
        for tid, t in self._tracked.items():
            if t["class_name"] != class_name:
                continue
            iou = self._iou(bbox, t["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_id = tid
        if best_iou >= self.track_iou_threshold:
            return best_id
        return None

    # ─── Snapshot ────────────────────────────────────────────────────────────

    def _save_snapshot(self, frame, label: str) -> Optional[str]:
        try:
            ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = self.alert_dir / f"{label}_{ts}.jpg"
            cv2.imwrite(str(path), frame)
            return str(path)
        except Exception as e:
            logger.error(f"snapshot save error: {e}")
            return None

    # ─── Main update loop ─────────────────────────────────────────────────────

    def update(
        self,
        frame,
        objects: list,
        people:  list,
        source_id: str = "",
        location:  str = "",
    ) -> List[dict]:
        """
        อัพเดทสถานะทุก frame
        คืนค่า list ของ alert dict เมื่อมีเหตุการณ์
        """
        alerts = []
        now    = time.time()
        seen_track_ids = set()

        for obj in objects:
            conf  = obj.get("confidence", 0.0)
            if conf < self.min_confidence:
                continue

            bbox       = obj.get("bbox", [0, 0, 0, 0])
            class_name = obj.get("class_name", "object")

            # [FIX] match to existing track via IoU instead of a
            # rounded-position key, so jitter doesn't reset first_seen
            tid = self._match_track(class_name, bbox)

            nearby = self._person_nearby(bbox, people)

            if tid is None:
                tid = self._next_track_id
                self._next_track_id += 1
                self._tracked[tid] = {
                    "first_seen":    now,
                    "last_seen":     now,
                    "bbox":          bbox,
                    "class_name":    class_name,
                    "confidence":    conf,
                    "had_person":    nearby,
                    "alert_sent":    False,
                    "missed":        0,
                }
            else:
                t = self._tracked[tid]
                t["last_seen"]  = now
                t["bbox"]       = bbox
                t["confidence"] = conf
                t["missed"]     = 0
                if nearby:
                    t["had_person"] = True

            seen_track_ids.add(tid)

            t = self._tracked[tid]
            elapsed = now - t["first_seen"]

            # ── ของถูกขโมย: เคยมีคนอยู่ใกล้ แล้วคนหายไป object ยังอยู่ ──
            if (t["had_person"] and not nearby
                    and elapsed >= self.theft_seconds
                    and not t["alert_sent"]):
                img_path = self._save_snapshot(frame, f"theft_{class_name}")
                alerts.append({
                    "event_type":         "object_theft",
                    "class_name":         class_name,
                    "confidence":         conf,
                    "bbox":               bbox,
                    "seconds_unattended": elapsed,
                    "source_id":          source_id,
                    "location":           location,
                    "image_path":         img_path,
                    "timestamp":          datetime.utcnow().isoformat(),
                    "alert_raised":       True,
                })
                t["alert_sent"] = True
                logger.info(f"[ObjectGuardian] THEFT detected: {class_name} @ {location}")

            # ── ของลืมทิ้ง: ไม่มีคนอยู่ใกล้ตลอด เกินเวลาที่กำหนด ──
            elif (not nearby
                    and not t["had_person"]
                    and elapsed >= self.left_seconds
                    and not t["alert_sent"]):
                img_path = self._save_snapshot(frame, f"left_{class_name}")
                alerts.append({
                    "event_type":         "object_left",
                    "class_name":         class_name,
                    "confidence":         conf,
                    "bbox":               bbox,
                    "seconds_unattended": elapsed,
                    "source_id":          source_id,
                    "location":           location,
                    "image_path":         img_path,
                    "timestamp":          datetime.utcnow().isoformat(),
                    "alert_raised":       True,
                })
                t["alert_sent"] = True
                logger.info(f"[ObjectGuardian] LEFT BEHIND detected: {class_name} @ {location}")

        # ── track ที่หายไปจาก frame นี้ ──
        # [FIX] อนุโลม track_max_missed เฟรมก่อนลบทิ้ง แทนที่จะลบทันที
        # ที่พลาดไปเฟรมเดียว (เช่น detection กระพริบ) ป้องกัน first_seen
        # โดน reset จากการ miss ชั่วคราว
        lost_ids = set(self._tracked.keys()) - seen_track_ids
        for tid in lost_ids:
            t = self._tracked[tid]
            t["missed"] += 1
            if t["missed"] > self.track_max_missed:
                del self._tracked[tid]

        return alerts

    # ─── Draw overlays ───────────────────────────────────────────────────────

    def draw(self, frame):
        for tid, t in self._tracked.items():
            x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
            color  = (0, 0, 255) if t["alert_sent"] else (0, 165, 255)
            label  = f"{t['class_name']} {int(time.time()-t['first_seen'])}s"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)