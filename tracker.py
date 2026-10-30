"""tracker.py — High-performance Hungarian Optimal Multi-Object Tracker (Linear Sum Assignment)
Replaces greedy tracker to eliminate track ID swapping, flapping, and lost identities during falls/posture changes.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import linear_sum_assignment


class HungarianTracker:
    """Optimal Linear Sum Assignment Tracker with two-stage IoU and Centroid matching.

    Key Features:
    1. Global Optimal Assignment: Resolves multiple simultaneous detections using Hungarian
       algorithm instead of first-come-first-served greedy matching.
    2. Two-Stage Cascaded Matching:
       - Stage 1: High IoU overlap matching for normal continuous movement.
       - Stage 2: Fall / Shape-Change Recovery matching using centroid distance and
         velocity prediction for remaining unmatched tracks (e.g., when a person falls
         and changes bounding box aspect ratio drastically from vertical to horizontal).
    3. Motion Prediction: Tracks velocity via exponential moving average (EMA) to predict
       target position during momentary occlusions or missed YOLO detections.
    4. Grace Period Memory: Preserves tracks for `max_lost` frames before retiring them,
       preventing re-ID churn.
    """

    def __init__(
        self,
        frame_w: int = 1920,
        frame_h: int = 1080,
        iou_thresh: float = 0.25,
        max_dist_norm: float = 0.15,
        max_lost: int = 30,
    ):
        self.next_id: int = 0
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.frame_w: int = frame_w
        self.frame_h: int = frame_h
        self._diag: float = float(math.sqrt(frame_w**2 + frame_h**2))
        self.iou_thresh: float = float(iou_thresh)
        self.max_dist_norm: float = float(max_dist_norm)
        self.max_lost: int = int(max_lost)

    @staticmethod
    def _box_center(box: List[float]) -> Tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @staticmethod
    def _box_wh(box: List[float]) -> Tuple[float, float]:
        return (box[2] - box[0], box[3] - box[1])

    @staticmethod
    def _iou(a: List[float], b: List[float]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        inter = w * h
        aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
        ab = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        union = aa + ab - inter
        return inter / union if union > 0 else 0.0

    def _dist_norm(self, c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
        d = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
        return d / self._diag if self._diag > 0 else 1.0

    def update(self, dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Updates tracks with current detections and assigns 'track_id' to each det."""
        if not dets:
            for tid in list(self.tracks):
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
            return []

        track_ids = list(self.tracks.keys())
        num_tracks = len(track_ids)
        num_dets = len(dets)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        # ── Stage 1: Hungarian IoU Matching for active tracks ───────────────
        if num_tracks > 0 and num_dets > 0:
            iou_cost = np.zeros((num_tracks, num_dets), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                t_box = self.tracks[tid]["bbox"]
                for j, det in enumerate(dets):
                    iou = self._iou(t_box, det["bbox"])
                    iou_cost[i, j] = 1.0 - iou  # Cost: 0.0 = perfect match, 1.0 = no overlap

            row_ind, col_ind = linear_sum_assignment(iou_cost)
            for r, c in zip(row_ind, col_ind):
                # Accept match if IoU >= threshold (cost <= 1.0 - thresh)
                if iou_cost[r, c] <= (1.0 - self.iou_thresh):
                    tid = track_ids[r]
                    matched_tracks.add(tid)
                    matched_dets.add(c)
                    self._update_track(tid, dets[c]["bbox"])
                    dets[c]["track_id"] = tid

        # ── Stage 2: Fall / Shape-Change Recovery via Centroid Matching ────
        # Handles posture collapse (standing -> lying) where IoU drops drastically
        unmatched_track_ids = [tid for tid in track_ids if tid not in matched_tracks]
        unmatched_det_indices = [j for j in range(num_dets) if j not in matched_dets]

        if unmatched_track_ids and unmatched_det_indices:
            dist_cost = np.zeros(
                (len(unmatched_track_ids), len(unmatched_det_indices)), dtype=np.float32
            )
            for i, tid in enumerate(unmatched_track_ids):
                t = self.tracks[tid]
                # Predict position using estimated velocity
                pred_cx = t["center"][0] + t["velocity"][0] * max(1, t["lost"])
                pred_cy = t["center"][1] + t["velocity"][1] * max(1, t["lost"])
                for j, d_idx in enumerate(unmatched_det_indices):
                    d_center = self._box_center(dets[d_idx]["bbox"])
                    dist_norm = self._dist_norm((pred_cx, pred_cy), d_center)
                    dx_norm = abs(pred_cx - d_center[0]) / max(1.0, float(self.frame_w))
                    # Gating threshold: Fall/collapse occurs in place (dx_norm <= 0.08)
                    # and overall movement is within max_dist_norm (0.10)
                    if dist_norm > min(0.10, self.max_dist_norm) or dx_norm > 0.08:
                        dist_cost[i, j] = 1000.0  # Prohibit match across distant people
                    else:
                        dist_cost[i, j] = dist_norm

            r_ind, c_ind = linear_sum_assignment(dist_cost)
            for r, c in zip(r_ind, c_ind):
                if dist_cost[r, c] <= self.max_dist_norm:
                    tid = unmatched_track_ids[r]
                    d_idx = unmatched_det_indices[c]
                    matched_tracks.add(tid)
                    matched_dets.add(d_idx)
                    self._update_track(tid, dets[d_idx]["bbox"])
                    dets[d_idx]["track_id"] = tid

        # ── Stage 3: New Tracks for unassigned detections ───────────────────
        for j in range(num_dets):
            if j not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                dets[j]["track_id"] = tid
                c = self._box_center(dets[j]["bbox"])
                self.tracks[tid] = {
                    "bbox": dets[j]["bbox"],
                    "center": c,
                    "velocity": (0.0, 0.0),
                    "lost": 0,
                    "age": 1,
                    "hits": 1,
                }

        # ── Stage 4: Maintain lost counter & prune expired tracks ───────────
        for tid in list(self.tracks):
            if tid not in matched_tracks:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        return dets

    def _update_track(self, tid: int, new_bbox: List[float]):
        old_c = self.tracks[tid]["center"]
        new_c = self._box_center(new_bbox)
        vx = new_c[0] - old_c[0]
        vy = new_c[1] - old_c[1]
        old_vx, old_vy = self.tracks[tid].get("velocity", (0.0, 0.0))
        # Exponential moving average for velocity
        smoothed_v = (0.7 * old_vx + 0.3 * vx, 0.7 * old_vy + 0.3 * vy)

        self.tracks[tid].update({
            "bbox": new_bbox,
            "center": new_c,
            "velocity": smoothed_v,
            "lost": 0,
            "age": self.tracks[tid].get("age", 0) + 1,
            "hits": self.tracks[tid].get("hits", 0) + 1,
        })

    def cleanup_track(self, tid: int):
        """Removes track state to prevent memory leaks."""
        self.tracks.pop(tid, None)


# Alias for backward compatibility
SimpleTracker = HungarianTracker


class SpatialAlertDeduplicator:
    """Suppresses duplicate alert triggers at the same physical camera location.

    Prevents alert storms when:
    1. YOLO produces two overlapping bounding boxes for the same fallen person.
    2. Tracker momentarily drops and re-acquires a person within a few seconds.
    """

    def __init__(self, radius_norm: float = 0.12, default_cooldown_sec: float = 30.0):
        self.radius_norm = radius_norm
        self.default_cooldown_sec = default_cooldown_sec
        # Dict of event_type -> list of (cx, cy, timestamp)
        self._history: Dict[str, List[Tuple[float, float, float]]] = {}

    def should_suppress(
        self,
        event_type: str,
        cx: float,
        cy: float,
        now_ts: float,
        cooldown_sec: Optional[float] = None,
    ) -> bool:
        cd = cooldown_sec if cooldown_sec is not None else self.default_cooldown_sec
        if event_type not in self._history:
            self._history[event_type] = []

        # Prune old records
        self._history[event_type] = [
            (x, y, t) for (x, y, t) in self._history[event_type] if (now_ts - t) < cd
        ]

        # Check proximity to recent alerts of this type
        for x, y, t in self._history[event_type]:
            dist = math.sqrt((cx - x) ** 2 + (cy - y) ** 2)
            if dist <= self.radius_norm:
                return True  # Suppress duplicate

        # Not duplicate -> record this alert
        self._history[event_type].append((cx, cy, now_ts))
        return False
