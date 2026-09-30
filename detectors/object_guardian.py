#!/usr/bin/env python3
"""
detectors/object_guardian.py
ตรวจจับของลืมทิ้ง (object left-behind) และของถูกขโมย (object theft)

[REFACTOR Steps 1 & 2]:
1. แก้ไขนิยามเหตุการณ์:
   - object_left: วัตถุยังอยู่ + เจ้าของไม่อยู่เกิน left_behind_seconds (นับจาก owner_left_at)
   - object_theft: วัตถุหายจากจุดเดิม + มีบุคคลอื่น (ไม่ใช่เจ้าของ) อยู่ใกล้จุดเกิดเหตุตอนของหาย
   - clean removal: วัตถุหายจากจุดเดิม โดยที่เจ้าของอยู่ใกล้ = เจ้าของเก็บของตัวเอง (ไม่ alert)
   - background: ของที่ไม่มีเจ้าของยืนยัน = ของในฉาก (ไม่ alert)
2. Owner Binding & Timer:
   - ผูกเจ้าของ (owner_tid) เมื่อคนอยู่ใกล้ต่อเนื่อง >= ownership_confirm_seconds
   - จับเวลา unattended จาก owner_left_at (เวลาที่เจ้าของออกห่าง) ไม่ใช่ first_seen
   - เจ้าของกลับมาใกล้ -> รีเซ็ต owner_left_at ทันที
3. Proximity Measurement (แทนที่ IoU เดิม):
   - สัญญาณข้อมือ (Wrist keypoints 9, 10): สัมผัส/หยิบจับใกล้ชิด
   - จุดศูนย์กลางวัตถุอยู่ภายใน Bbox ของคน (Center containment)
   - Normalized Edge-to-Edge Distance เทียบกับความสูงคน <= proximity_norm
"""
import cv2
import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ObjectGuardian:
    def __init__(self, cfg: dict):
        self.cfg                = cfg
        self.alert_dir          = Path(cfg.get("alert_dir", "alerts"))
        self.alert_dir.mkdir(parents=True, exist_ok=True)

        # Thresholds
        self.left_seconds       = float(cfg.get("left_behind_seconds", 15))
        self.theft_seconds      = float(cfg.get("theft_window_seconds", cfg.get("theft_seconds", 8)))
        self.ownership_confirm_seconds = float(cfg.get("ownership_confirm_seconds", 2.0))
        self.proximity_norm     = float(cfg.get("proximity_norm", 0.20))
        self.min_confidence     = float(cfg.get("min_confidence", 0.25))
        self.track_iou_threshold = float(cfg.get("track_iou_threshold", 0.15))
        self.track_max_missed   = int(cfg.get("track_max_missed", 15))
        self.missing_confirm_frames = int(cfg.get("missing_confirm_frames", 3))
        self.cooldown_seconds   = float(cfg.get("cooldown_seconds", 60))

        # Classes to ignore — static furniture that triggers false alerts
        _default_ignore = {
            "chair", "dining table", "bench", "couch", "bed",
            "toilet", "oven", "tv", "sink", "refrigerator",
            "parking meter", "fire hydrant", "stop sign",
            "traffic light", "bird",
        }
        self.ignore_classes = set(
            cfg.get("ignore_classes", _default_ignore)
        )

        # State tracking: track_id -> dict
        self._tracked: Dict[int, dict] = {}
        self._next_track_id = 0

    # ─── Proximity & Geometry ────────────────────────────────────────────────

    def _iou(self, a, b) -> float:
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        w  = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
        inter = w * h
        aa = max(1e-6, (a[2]-a[0])*(a[3]-a[1]))
        ab = max(1e-6, (b[2]-b[0])*(b[3]-b[1]))
        return inter / (aa + ab - inter)

    def _is_person_near(self, obj_bbox: list, person: dict, proximity_norm: float = None) -> Tuple[bool, str, float]:
        """
        ตรวจสอบว่า person อยู่ใกล้ obj_bbox หรือไม่ ด้วย 3 สัญญาณ:
        1. Wrist Keypoint (สัมผัส/หยิบจับ)
        2. Center Containment (วัตถุอยู่ภายในตัวคน)
        3. Normalized Edge Distance (ระยะห่างขอบต่อขอบ เทียบความสูงคน)
        คืนค่า: (is_near, reason, distance_score)
        """
        p_norm = proximity_norm if proximity_norm is not None else self.proximity_norm
        p_bbox = person.get("bbox")
        if not p_bbox or len(p_bbox) < 4:
            return False, "none", 999.0

        p_h = max(10.0, float(p_bbox[3] - p_bbox[1]))

        # สัญญาณที่ 1: Wrist keypoints (Joint 9: left wrist, Joint 10: right wrist)
        kpts = person.get("keypoints")
        if kpts and len(kpts) > 10:
            pad_x = max(20.0, 0.25 * (obj_bbox[2] - obj_bbox[0]))
            pad_y = max(20.0, 0.25 * (obj_bbox[3] - obj_bbox[1]))
            for w_idx in (9, 10):
                kp = kpts[w_idx]
                if len(kp) >= 3:
                    wx, wy, c = kp[0], kp[1], kp[2]
                    if c >= 0.30:
                        if (obj_bbox[0] - pad_x <= wx <= obj_bbox[2] + pad_x and
                            obj_bbox[1] - pad_y <= wy <= obj_bbox[3] + pad_y):
                            return True, "wrist", 0.0

        # สัญญาณที่ 2: Center containment (จุดศูนย์กลางของวัตถุอยู่ข้างในกรอบตัวคน)
        ox = (obj_bbox[0] + obj_bbox[2]) / 2.0
        oy = (obj_bbox[1] + obj_bbox[3]) / 2.0
        if (p_bbox[0] <= ox <= p_bbox[2] and p_bbox[1] <= oy <= p_bbox[3]):
            return True, "containment", 0.0

        # สัญญาณที่ 3: Edge-to-edge distance เทียบกับความสูงของคน
        dx = max(0.0, max(p_bbox[0] - obj_bbox[2], obj_bbox[0] - p_bbox[2]))
        dy = max(0.0, max(p_bbox[1] - obj_bbox[3], obj_bbox[1] - p_bbox[3]))
        edge_dist = (dx * dx + dy * dy) ** 0.5
        norm_dist = edge_dist / p_h

        if norm_dist <= p_norm:
            return True, "proximity", norm_dist

        return False, "none", norm_dist

    def _find_nearby_people(self, obj_bbox: list, people: list, proximity_norm: float = None) -> List[Tuple[dict, str, float]]:
        """
        ค้นหาคนทั้งหมดที่อยู่ใกล้ obj_bbox จัดเรียงตามความใกล้ชิด
        คืนค่า list of (person_dict, reason, distance_score)
        """
        nearby = []
        for p in people:
            is_near, reason, dist = self._is_person_near(obj_bbox, p, proximity_norm=proximity_norm)
            if is_near:
                nearby.append((p, reason, dist))
        nearby.sort(key=lambda item: item[2])
        return nearby

    def _match_track(self, class_name: str, bbox) -> Optional[int]:
        """จับคู่ track เดิมด้วย IoU ข้ามเฟรม"""
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
        save_local = os.getenv("SAVE_LOCAL_SNAPSHOTS", "false").lower() == "true"
        if not save_local:
            try:
                from utils import frame_to_base64
                return frame_to_base64(frame)
            except Exception as e:
                logger.error(f"Base64 snapshot error: {e}")
                return None
        try:
            ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = self.alert_dir / f"{label}_{ts}.jpg"
            cv2.imwrite(str(path), frame)
            return str(path)
        except Exception as e:
            logger.error(f"Snapshot save error: {e}")
            return None

    # ─── Main Update Loop ────────────────────────────────────────────────────

    def update(
        self,
        frame,
        objects: list,
        people:  list,
        source_id: str = "",
        location:  str = "",
        t_sec: float = None,
    ) -> List[dict]:
        """
        อัปเดตสถานะทุกเฟรม:
        people: list of dict จาก SimpleTracker (ต้องมี 'track_id', 'bbox', และ 'keypoints' สำหรับมือ)
        objects: list of dict จาก model_server ('bbox', 'class_name', 'confidence')
        """
        alerts = []
        now = t_sec if t_sec is not None else time.time()
        seen_track_ids = set()

        # ── 1. ประมวลผลวัตถุที่ตรวจจับพบในเฟรม ──
        for obj in objects:
            conf = obj.get("confidence", obj.get("conf", 0.0))
            if conf < self.min_confidence:
                continue

            bbox = obj.get("bbox", [0, 0, 0, 0])
            class_name = obj.get("class_name", "object")

            if class_name in self.ignore_classes:
                continue

            tid = self._match_track(class_name, bbox)
            nearby_people = self._find_nearby_people(bbox, people)

            if tid is None:
                tid = self._next_track_id
                self._next_track_id += 1
                self._tracked[tid] = {
                    "track_id":              tid,
                    "first_seen":            now,
                    "last_seen":             now,
                    "bbox":                  bbox,
                    "class_name":            class_name,
                    "confidence":            conf,
                    "state":                 "NEW",
                    "owner_tid":             None,
                    "candidate_owner_tid":   None,
                    "candidate_first_near":  None,
                    "owner_left_at":         None,
                    "alert_sent":            False,
                    "alert_type":            None,
                    "missed":                0,
                    "recent_nearby_people":  [],
                }
            else:
                t = self._tracked[tid]
                t["last_seen"]  = now
                t["bbox"]       = bbox
                t["confidence"] = conf
                t["missed"]     = 0

            seen_track_ids.add(tid)
            t = self._tracked[tid]

            # บันทึกคนใกล้ชิดในเฟรมปัจจุบันไว้ตรวจสอบย้อนหลังเมื่อของหาย
            current_near_summary = [
                {"track_id": p[0].get("track_id"), "reason": p[1], "dist": p[2]}
                for p in nearby_people if p[0].get("track_id") is not None
            ]
            t["recent_nearby_people"] = current_near_summary

            # ── [Owner Binding & Left Detection Logic] ──
            if t["owner_tid"] is None:
                # วัตถุยังไม่มีเจ้าของ: หาผู้มีแนวโน้มเป็นเจ้าของ (Candidate)
                if nearby_people:
                    best_person, best_reason, _ = nearby_people[0]
                    cand_tid = best_person.get("track_id")
                    if cand_tid is not None:
                        if t["candidate_owner_tid"] == cand_tid:
                            # คนเดิมอยู่ใกล้ต่อเนื่อง
                            cand_duration = now - t["candidate_first_near"]
                            if cand_duration >= self.ownership_confirm_seconds:
                                t["owner_tid"] = cand_tid
                                t["state"] = "OWNED"
                                t["owner_left_at"] = None
                                logger.info(
                                    f"[ObjectGuardian] Object #{tid} ({class_name}) "
                                    f"bound to owner person_id={cand_tid} ({best_reason})"
                                )
                        else:
                            t["candidate_owner_tid"] = cand_tid
                            t["candidate_first_near"] = now
                else:
                    t["candidate_owner_tid"] = None
                    t["candidate_first_near"] = None

            else:
                # วัตถุมีเจ้าของแล้ว: ตรวจสอบสถานะการอยู่ใกล้ของเจ้าของ
                owner_is_near = any(
                    p[0].get("track_id") == t["owner_tid"] for p in nearby_people
                )

                if owner_is_near:
                    # เจ้าของยังอยู่ใกล้ หรือกลับมาใกล้ -> รีเซ็ตเวลา owner_left_at
                    t["owner_left_at"] = None
                    t["state"] = "OWNED"
                else:
                    # เจ้าของไม่อยู่ใกล้
                    if t["owner_left_at"] is None:
                        t["owner_left_at"] = now
                    t["state"] = "UNATTENDED"

                    unattended_elapsed = now - t["owner_left_at"]

                    # ── เหตุการณ์: ของลืมทิ้ง (object_left) ──
                    if (unattended_elapsed >= self.left_seconds
                            and not t["alert_sent"]):
                        img_path = self._save_snapshot(frame, f"left_{class_name}")
                        alert_dict = {
                            "event_type":         "object_left",
                            "class_name":         class_name,
                            "confidence":         conf,
                            "bbox":               bbox,
                            "seconds_unattended": unattended_elapsed,
                            "source_id":          source_id,
                            "location":           location,
                            "image_path":         img_path,
                            "timestamp":          datetime.utcnow().isoformat(),
                            "alert_raised":       True,
                            "track_id":           tid,
                            "object_id":          tid,
                            "owner_track_id":     t["owner_tid"],
                            "suspect_track_id":   None,
                            "person_track_id":    t["owner_tid"],
                        }
                        alerts.append(alert_dict)
                        t["alert_sent"] = True
                        t["alert_type"] = "object_left"
                        logger.info(
                            f"[ObjectGuardian] LEFT BEHIND detected: {class_name} "
                            f"(owner {t['owner_tid']} left for {unattended_elapsed:.1f}s) @ {location}"
                        )

        # ── 2. ตรวจสอบวัตถุที่หายไปจากเฟรม (Lost / Theft Detection) ──
        lost_ids = set(self._tracked.keys()) - seen_track_ids
        for tid in lost_ids:
            t = self._tracked[tid]
            t["missed"] += 1

            # ถ้าวัตถุเคยมีเจ้าของและยังไม่เคยส่ง alert
            if t.get("owner_tid") is not None and not t["alert_sent"]:
                # เมื่อหายไปต่อเนื่องอย่างน้อย missing_confirm_frames เพื่อกรองเฟรมกะพริบ
                if t["missed"] >= self.missing_confirm_frames:
                    # ตรวจสอบคนใกล้ชิดในเฟรมปัจจุบันที่ตำแหน่งของวัตถุ
                    nearby_curr = self._find_nearby_people(t["bbox"], people)
                    curr_tids = {p[0].get("track_id") for p in nearby_curr if p[0].get("track_id") is not None}

                    # และตรวจสอบคนใกล้ชิดในเฟรมล่าสุดก่อนที่วัตถุจะหายไป
                    recent_tids = {p["track_id"] for p in t.get("recent_nearby_people", []) if p.get("track_id") is not None}

                    combined_near_tids = curr_tids | recent_tids

                    owner_id = t["owner_tid"]
                    if owner_id in combined_near_tids:
                        # เจ้าของเป็นผู้เก็บของไปเอง -> สะอาด ไม่ alert
                        t["state"] = "RETRIEVED_BY_OWNER"
                        t["alert_sent"] = True
                        logger.info(f"[ObjectGuardian] Object #{tid} ({t['class_name']}) retrieved by owner {owner_id}")
                    else:
                        # หาว่ามีบุคคลอื่น (ไม่ใช่เจ้าของ) อยู่ใกล้จุดเกิดเหตุหรือไม่
                        other_tids = [pid for pid in combined_near_tids if pid != owner_id]
                        if other_tids:
                            # ── เหตุการณ์: ของถูกขโมย (object_theft) ──
                            suspect_id = other_tids[0]
                            img_path = self._save_snapshot(frame, f"theft_{t['class_name']}")
                            elapsed_unattended = (now - t["owner_left_at"]) if t.get("owner_left_at") else 0.0

                            alert_dict = {
                                "event_type":         "object_theft",
                                "class_name":         t["class_name"],
                                "confidence":         t["confidence"],
                                "bbox":               t["bbox"],
                                "seconds_unattended": elapsed_unattended,
                                "source_id":          source_id,
                                "location":           location,
                                "image_path":         img_path,
                                "timestamp":          datetime.utcnow().isoformat(),
                                "alert_raised":       True,
                                "track_id":           tid,
                                "object_id":          tid,
                                "owner_track_id":     owner_id,
                                "suspect_track_id":   suspect_id,
                                "person_track_id":    suspect_id,
                            }
                            alerts.append(alert_dict)
                            t["state"] = "STOLEN"
                            t["alert_sent"] = True
                            t["alert_type"] = "object_theft"
                            logger.info(
                                f"[ObjectGuardian] THEFT detected: {t['class_name']} "
                                f"stolen by suspect {suspect_id} (owner: {owner_id}) @ {location}"
                            )

            # ลบ Track ที่หายไปนานเกินกำหนดออกจากหน่วยความจำ
            if t["missed"] > self.track_max_missed:
                del self._tracked[tid]

        return alerts

    # ─── Draw Overlays ───────────────────────────────────────────────────────

    def draw(self, frame):
        for tid, t in self._tracked.items():
            x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
            state = t.get("state", "NEW")
            owner = t.get("owner_tid")

            if t["alert_sent"]:
                color = (0, 0, 255)        # แดง: Alert แล้ว
            elif state == "UNATTENDED":
                color = (0, 165, 255)      # ส้ม: ไม่มีคนเฝ้า
            elif state == "OWNED":
                color = (0, 255, 0)        # เขียว: มีเจ้าของอยู่ใกล้
            else:
                color = (200, 200, 200)    # เทา: ของใหม่/ยังไม่มีเจ้าของ

            owner_str = f"owner:{owner}" if owner is not None else "no-owner"
            label = f"{t['class_name']} #{tid} [{owner_str}] {state}"

            if state == "UNATTENDED" and t.get("owner_left_at"):
                unattended_s = int(time.time() - t["owner_left_at"])
                label += f" {unattended_s}s/{int(self.left_seconds)}s"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(12, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)