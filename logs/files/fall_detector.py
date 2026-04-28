"""
=============================================================
 Dangerous Fall Detector  –  YOLOv8-Pose + Temporal Analysis
=============================================================
Architecture:
  1. YOLOv8-Pose  → skeleton keypoints per person
  2. Physics Metrics  → angle, velocity, aspect ratio
  3. Temporal Buffer  → confirm fall is SUSTAINED (not a stumble)
  4. Danger Classifier → rule-based + optional ML scorer

Key design choices to minimise false positives
  • Stumble: fast, body recovers → short HIGH angle duration
  • Dangerous fall: sustained horizontal + head near floor + high descent velocity
"""

import cv2
import numpy as np
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import json
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s')
logger = logging.getLogger(__name__)

# ─── YOLO import (graceful degradation) ──────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed.  Install with: pip install ultralytics")


# ─────────────────────────────────────────────────────────────────────────────
# Keypoint indices  (COCO 17-point skeleton)
# ─────────────────────────────────────────────────────────────────────────────
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


# ─────────────────────────────────────────────────────────────────────────────
# Tuneable thresholds  (edit here or load from configs/thresholds.yaml)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FallThresholds:
    # --- Pose metrics ---
    torso_angle_deg: float = 50.0      # body tilted beyond X° from vertical
    bbox_aspect_ratio: float = 0.8     # w/h ratio  (>1 = person is lying down)
    head_below_hip_factor: float = 0.2 # head y > hip_y + factor*person_height
    min_keypoint_conf: float = 0.3     # ignore low-conf keypoints

    # --- Velocity ---
    fall_velocity_px_per_frame: float = 8.0   # downward speed of centre-of-mass

    # --- Temporal (anti-stumble filter) ---
    fall_confirm_frames: int = 8        # consecutive "fall pose" frames needed
    fall_clear_frames: int = 20         # frames of normal pose to clear alert
    min_fall_duration_sec: float = 0.8  # ignore events shorter than this

    # --- Danger score ---
    danger_score_threshold: float = 0.60   # 0-1 final score to raise alert


@dataclass
class PersonState:
    """Rolling state for one tracked person."""
    track_id: int
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=30))
    pose_history: deque = field(default_factory=lambda: deque(maxlen=30))
    fall_frame_count: int = 0
    normal_frame_count: int = 0
    alert_active: bool = False
    fall_start_time: Optional[float] = None
    last_center: Optional[np.ndarray] = None
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=10))


# ─────────────────────────────────────────────────────────────────────────────
# Core physics / pose metrics
# ─────────────────────────────────────────────────────────────────────────────

def _midpoint(a, b):
    return (a + b) / 2.0


def compute_torso_angle(kps: np.ndarray) -> Optional[float]:
    """
    Angle of the torso from vertical (0° = standing, 90° = lying).
    Uses shoulder-mid → hip-mid vector.
    """
    ls, rs = kps[KP["left_shoulder"]], kps[KP["right_shoulder"]]
    lh, rh = kps[KP["left_hip"]], kps[KP["right_hip"]]

    if min(ls[2], rs[2], lh[2], rh[2]) < 0.3:
        return None

    shoulder_mid = _midpoint(ls[:2], rs[:2])
    hip_mid      = _midpoint(lh[:2], rh[:2])

    vec = hip_mid - shoulder_mid
    # angle from vertical  (vec pointing downward = 0°)
    angle = np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6))
    return float(angle)


def compute_bbox_aspect_ratio(bbox: np.ndarray) -> float:
    """w/h of bounding box.  >1 means person occupies horizontal space."""
    x1, y1, x2, y2 = bbox
    w, h = abs(x2 - x1) + 1e-6, abs(y2 - y1) + 1e-6
    return float(w / h)


def is_head_below_hips(kps: np.ndarray, person_height: float) -> bool:
    """True if the nose/head is significantly below the hip centre."""
    nose = kps[KP["nose"]]
    lh, rh = kps[KP["left_hip"]], kps[KP["right_hip"]]

    if nose[2] < 0.3 or min(lh[2], rh[2]) < 0.3:
        return False

    hip_mid_y = _midpoint(lh[:2], rh[:2])[1]
    # In image coords y increases downward
    return bool(nose[1] > hip_mid_y + 0.1 * person_height)


def compute_danger_score(
    torso_angle: Optional[float],
    aspect_ratio: float,
    head_below: bool,
    velocity_y: float,
    thresholds: FallThresholds,
) -> float:
    """
    Weighted danger score  ∈ [0, 1].
    Higher = more likely a real dangerous fall.
    """
    score = 0.0

    # 1. Torso angle  (weight 0.35)
    if torso_angle is not None:
        angle_norm = min(torso_angle / 90.0, 1.0)
        if torso_angle >= thresholds.torso_angle_deg:
            score += 0.35 * angle_norm

    # 2. Bounding-box aspect ratio  (weight 0.25)
    if aspect_ratio >= thresholds.bbox_aspect_ratio:
        ar_norm = min((aspect_ratio - 0.8) / 1.2, 1.0)
        score += 0.25 * ar_norm

    # 3. Head below hips  (weight 0.25)
    if head_below:
        score += 0.25

    # 4. Downward velocity  (weight 0.15)
    vel_norm = min(velocity_y / thresholds.fall_velocity_px_per_frame, 1.0)
    if vel_norm > 0:
        score += 0.15 * vel_norm

    return min(score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Main detector class
# ─────────────────────────────────────────────────────────────────────────────

class DangerousFallDetector:
    """
    Drop-in detector.  Call  .process_frame(frame)  per video frame.
    Returns annotated frame + list of FallEvent dicts.
    """

    def __init__(
        self,
        model_path: str = "yolov8l-pose.pt",
        thresholds: Optional[FallThresholds] = None,
        device: str = "cpu",
    ):
        self.thresholds = thresholds or FallThresholds()
        self.person_states: dict[int, PersonState] = {}
        self.frame_idx = 0
        self.fps_estimate = 30.0

        if YOLO_AVAILABLE:
            logger.info(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            self.model.to(device)
        else:
            self.model = None
            logger.warning("Running without YOLO model (demo mode)")

    # ── public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray):
        """
        Parameters
        ----------
        frame : BGR image (H, W, 3)

        Returns
        -------
        annotated_frame : BGR image with overlays
        events          : list of dicts  {track_id, danger_score, bbox, ...}
        """
        self.frame_idx += 1
        events = []

        if self.model is None:
            return frame, events

        results = self.model.track(
            frame,
            persist=True,
            conf=0.35,
            iou=0.45,
            classes=[0],          # persons only
            verbose=False,
        )

        if results and results[0].boxes is not None:
            boxes  = results[0].boxes
            keypoints_all = results[0].keypoints

            ids = boxes.id
            if ids is None:
                return frame, events

            for i, track_id in enumerate(ids.int().tolist()):
                bbox = boxes.xyxy[i].cpu().numpy()
                kps  = keypoints_all.data[i].cpu().numpy()  # (17, 3)

                state = self.person_states.setdefault(
                    track_id, PersonState(track_id=track_id)
                )

                event = self._analyse_person(state, bbox, kps, frame)
                if event:
                    events.append(event)

            frame = self._draw_overlays(frame, results[0], events)

        return frame, events

    # ── internal helpers ──────────────────────────────────────────────────────

    def _analyse_person(
        self,
        state: PersonState,
        bbox: np.ndarray,
        kps: np.ndarray,
        frame: np.ndarray,
    ) -> Optional[dict]:
        """Return a fall event dict if a dangerous fall is confirmed, else None."""

        th = self.thresholds
        now = time.time()

        # ── compute metrics ──
        torso_angle  = compute_torso_angle(kps)
        aspect_ratio = compute_bbox_aspect_ratio(bbox)
        person_h     = abs(bbox[3] - bbox[1]) + 1e-6
        head_below   = is_head_below_hips(kps, person_h)

        # velocity of bounding-box centre
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        centre = np.array([cx, cy])

        velocity_y = 0.0
        if state.last_center is not None:
            velocity_y = max(0.0, float(centre[1] - state.last_center[1]))
        state.last_center = centre
        state.velocity_history.append(velocity_y)
        avg_velocity_y = float(np.mean(state.velocity_history))

        # ── danger score ──
        score = compute_danger_score(
            torso_angle, aspect_ratio, head_below, avg_velocity_y, th
        )

        is_fall_pose = score >= th.danger_score_threshold

        # ── temporal confirmation (anti-stumble filter) ──
        if is_fall_pose:
            state.fall_frame_count  += 1
            state.normal_frame_count = 0
        else:
            state.normal_frame_count += 1
            if state.normal_frame_count >= th.fall_clear_frames:
                state.fall_frame_count = 0
                if state.alert_active:
                    logger.info(f"[Track {state.track_id}] Fall cleared")
                state.alert_active = False
                state.fall_start_time = None

        # ── raise alert ──
        if (state.fall_frame_count >= th.fall_confirm_frames
                and not state.alert_active):
            state.alert_active   = True
            state.fall_start_time = now
            logger.warning(
                f"⚠️  FALL ALERT  track={state.track_id}  score={score:.2f}  "
                f"angle={torso_angle}  ar={aspect_ratio:.2f}"
            )

        if state.alert_active:
            duration = now - (state.fall_start_time or now)
            return {
                "track_id":     state.track_id,
                "danger_score": round(score, 3),
                "duration_sec": round(duration, 2),
                "torso_angle":  round(torso_angle, 1) if torso_angle else None,
                "aspect_ratio": round(aspect_ratio, 3),
                "head_below_hips": head_below,
                "velocity_y":   round(avg_velocity_y, 2),
                "bbox":         bbox.tolist(),
                "frame":        self.frame_idx,
            }

        return None

    def _draw_overlays(self, frame, result, events: list) -> np.ndarray:
        """Draw skeleton + alert boxes on frame."""
        alert_ids = {e["track_id"] for e in events}

        # draw skeleton
        annotated = result.plot(
            boxes=True,
            labels=True,
            conf=True,
            kpt_radius=4,
            line_width=2,
        )

        # overlay alert for falling persons
        for event in events:
            x1, y1, x2, y2 = [int(v) for v in event["bbox"]]
            score = event["danger_score"]
            label = f"⚠ FALL  {score:.0%}  {event['duration_sec']:.1f}s"

            # red semi-transparent rect
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 220), -1)
            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # label banner
            (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - th_ - 10), (x1 + tw + 8, y1),
                          (0, 0, 200), -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

        # HUD
        cv2.putText(annotated, f"Frame {self.frame_idx} | Falls: {len(events)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 100), 2)

        return annotated


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def run_on_video(video_path: str, output_path: str = "output_fall.mp4",
                 model: str = "yolov8l-pose.pt", device: str = "cpu"):
    detector = DangerousFallDetector(model_path=model, device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out  = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    detector.fps_estimate = fps
    all_events = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, events = detector.process_frame(frame)
        all_events.extend(events)
        out.write(annotated)

    cap.release()
    out.release()

    # save event log
    log_path = output_path.replace(".mp4", "_events.json")
    with open(log_path, "w") as f:
        json.dump(all_events, f, indent=2, default=str)

    logger.info(f"Done.  Output: {output_path}  Events: {log_path}")
    return all_events


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fall_detector.py <video_path> [output.mp4] [model.pt]")
        sys.exit(1)
    video  = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "output_fall.mp4"
    mdl    = sys.argv[3] if len(sys.argv) > 3 else "yolov8l-pose.pt"
    run_on_video(video, output, mdl)
