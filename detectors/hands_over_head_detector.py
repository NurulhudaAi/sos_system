import time
import logging
import numpy as np

logger = logging.getLogger("hands_over_head_detector")

# COCO 17 keypoint indices
NOSE        = 0
L_EYE       = 1
R_EYE       = 2
L_SHOULDER  = 5
R_SHOULDER  = 6
L_ELBOW     = 7
R_ELBOW     = 8
L_WRIST     = 9
R_WRIST     = 10
L_HIP       = 11
R_HIP       = 12


def _kp(kps, i):
    return kps[i] if i < len(kps) else np.zeros(3)


def _vis(kp, conf=0.3):
    return float(kp[2]) >= conf


class HandsOverHeadDetector:
    """Detect SOS gesture: both hands raised above head (pose_sos).

    Uses YOLO pose keypoints (COCO-17) — no MediaPipe needed.

    Detection logic:
      1. Both wrists must be visible (confidence ≥ min_kp_conf)
      2. Both wrists Y-coordinate must be ABOVE nose AND shoulders
         (with configurable margin)
      3. Body must be upright (not lying down / fallen):
         shoulders must be above hips in image plane
      4. Wrists must be at or above elbows
      5. Must be sustained for ≥ confirm_seconds to avoid false positives
         from temporary arm movements (e.g. stretching, waving)

    Config keys (thresholds.yaml → hands_over_head:):
      min_kp_conf        0.3    minimum keypoint confidence
      confirm_seconds    3.0    sustained duration before trigger
      cooldown_seconds   60     cooldown between consecutive triggers
      margin_above_nose  0.03   wrists must be this fraction of frame height
                                above the reference point (nose/shoulder)
    """

    def __init__(self, cfg: dict):
        self.min_kp_conf      = cfg.get("min_kp_conf", 0.3)
        self.confirm_s        = cfg.get("confirm_seconds", 3.0)
        self.cooldown_s       = cfg.get("cooldown_seconds", 60)
        self.margin_above     = cfg.get("margin_above_nose", 0.03)

        # Per-track state
        self._since = {}       # when gesture first detected
        self._cooldown = {}    # last trigger time

    def cleanup_track(self, tid: int) -> None:
        """Remove per-track state for a disappeared track."""
        self._since.pop(tid, None)
        self._cooldown.pop(tid, None)

    def process(self, tid: int, kps, h: int, w: int, timestamp: float = None) -> dict:
        """Process one person's keypoints for hands-over-head gesture.

        Parameters
        ----------
        tid       : track ID
        kps       : COCO-17 keypoints [[x, y, conf], ...]
        h, w      : frame dimensions
        timestamp : video time in seconds (optional, defaults to time.time())

        Returns
        -------
        dict with keys:
          gesture_detected : bool — raw per-frame detection
          is_confirmed     : bool — sustained for confirm_seconds
          time_held        : float — how long gesture has been held (seconds)
          triggered        : bool — confirmed AND not in cooldown
        """
        now = timestamp if timestamp is not None else time.time()
        h_pixels = float(h) if h else 1.0
        margin_px = self.margin_above * h_pixels

        # Get keypoints
        nose = _kp(kps, NOSE)
        l_shoulder = _kp(kps, L_SHOULDER)
        r_shoulder = _kp(kps, R_SHOULDER)
        l_elbow = _kp(kps, L_ELBOW)
        r_elbow = _kp(kps, R_ELBOW)
        l_wrist = _kp(kps, L_WRIST)
        r_wrist = _kp(kps, R_WRIST)
        l_hip = _kp(kps, L_HIP)
        r_hip = _kp(kps, R_HIP)

        # Check visibility — need both wrists + at least one reference point
        wrists_visible = _vis(l_wrist, self.min_kp_conf) and _vis(r_wrist, self.min_kp_conf)
        has_reference = (_vis(nose, self.min_kp_conf) or
                        (_vis(l_shoulder, self.min_kp_conf) and _vis(r_shoulder, self.min_kp_conf)))

        if not wrists_visible or not has_reference:
            self._since.pop(tid, None)
            return {
                "gesture_detected": False,
                "is_confirmed": False,
                "time_held": 0.0,
                "triggered": False,
            }

        # Determine reference Y (higher/lower number = lower in image)
        # Use nose if visible, otherwise shoulder midpoint
        if _vis(nose, self.min_kp_conf):
            ref_y = float(nose[1])
        else:
            ref_y = (float(l_shoulder[1]) + float(r_shoulder[1])) / 2

        # Also check shoulders — wrists should be above shoulders too
        shoulder_y = float('inf')
        if _vis(l_shoulder, self.min_kp_conf) and _vis(r_shoulder, self.min_kp_conf):
            shoulder_y = min(float(l_shoulder[1]), float(r_shoulder[1]))

        # Both wrists must be ABOVE reference (lower Y value = higher in image)
        l_wrist_y = float(l_wrist[1])
        r_wrist_y = float(r_wrist[1])

        gesture_detected = (
            l_wrist_y < ref_y - margin_px and
            r_wrist_y < ref_y - margin_px and
            l_wrist_y < shoulder_y and
            r_wrist_y < shoulder_y
        )

        # ── Check Upright Torso ──
        # If hips are visible, ensure person is upright (not lying down / prone)
        has_hips = _vis(l_hip, self.min_kp_conf) or _vis(r_hip, self.min_kp_conf)
        if has_hips:
            hips_y = []
            if _vis(l_hip, self.min_kp_conf): hips_y.append(float(l_hip[1]))
            if _vis(r_hip, self.min_kp_conf): hips_y.append(float(r_hip[1]))
            avg_hip_y = sum(hips_y) / len(hips_y)
            # Reference/shoulders must be well above hips (lower Y in image)
            if ref_y >= avg_hip_y - margin_px:
                gesture_detected = False

        # ── Check Elbows ──
        # Wrists should be at or above elbows when elbows are visible
        if _vis(l_elbow, self.min_kp_conf) and l_wrist_y > float(l_elbow[1]) + margin_px:
            gesture_detected = False
        if _vis(r_elbow, self.min_kp_conf) and r_wrist_y > float(r_elbow[1]) + margin_px:
            gesture_detected = False

        # Temporal confirmation
        if gesture_detected:
            self._since.setdefault(tid, now)
            time_held = now - self._since[tid]
        else:
            self._since.pop(tid, None)
            time_held = 0.0

        is_confirmed = gesture_detected and time_held >= self.confirm_s

        # Cooldown check
        triggered = False
        if is_confirmed:
            last_trigger = self._cooldown.get(tid, -1e9)
            if now - last_trigger >= self.cooldown_s:
                triggered = True
                self._cooldown[tid] = now

        return {
            "gesture_detected": bool(gesture_detected),
            "is_confirmed": bool(is_confirmed),
            "time_held": round(time_held, 1),
            "triggered": bool(triggered),
        }
