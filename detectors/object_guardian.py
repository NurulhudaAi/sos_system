#!/usr/bin/env python3
"""
detectors/object_guardian.py
ตรวจจับของลืมทิ้ง (object left-behind) และของถูกขโมย (object theft)
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

        # state
        self._tracked: Dict[str, dict] = {}   # object_key → state
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
        seen_keys = set()

        for obj in objects:
            conf  = obj.get("confidence", 0.0)
            if conf < self.min_confidence:
                continue

            bbox       = obj.get("bbox", [0, 0, 0, 0])
            class_name = obj.get("class_name", "object")
            key        = f"{class_name}_{int(bbox[0])}_{int(bbox[1])}"
            seen_keys.add(key)

            nearby = self._person_nearby(bbox, people)

            if key not in self._tracked:
                self._tracked[key] = {
                    "first_seen":    now,
                    "last_seen":     now,
                    "bbox":          bbox,
                    "class_name":    class_name,
                    "confidence":    conf,
                    "had_person":    nearby,
                    "alert_sent":    False,
                }
            else:
                t = self._tracked[key]
                t["last_seen"]  = now
                t["bbox"]       = bbox
                t["confidence"] = conf
                if nearby:
                    t["had_person"] = True

            t = self._tracked[key]
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

        # ── ลบ object ที่หายจาก frame ──
        lost_keys = set(self._tracked.keys()) - seen_keys
        for k in lost_keys:
            del self._tracked[k]

        return alerts

    # ─── Draw overlays ───────────────────────────────────────────────────────

    def draw(self, frame):
        for key, t in self._tracked.items():
            x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
            color  = (0, 0, 255) if t["alert_sent"] else (0, 165, 255)
            label  = f"{t['class_name']} {int(time.time()-t['first_seen'])}s"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
