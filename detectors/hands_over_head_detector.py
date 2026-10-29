
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
    """Detect SOS gesture:
      Option 1: Triangle gesture above head (wrists meet together with elbows out forming a roof).
      Option 2: Both hands raised above head in V-shape.

    Uses YOLO pose keypoints (COCO-17) — no MediaPipe needed.

    Config keys (thresholds.yaml → hands_over_head:):
      min_kp_conf        0.3    minimum keypoint confidence
      confirm_seconds    7.0    sustained duration before trigger (7-10s timeline)
      cooldown_seconds   30     cooldown between consecutive triggers
      margin_above_nose  0.03   wrists must be this fraction of frame height
                                above the reference point (nose/shoulder)
    """

    def __init__(self, cfg: dict):
        self.min_kp_conf      = cfg.get("min_kp_conf", 0.3)
        self.confirm_s        = cfg.get("confirm_seconds", 7.0)
        self.cooldown_s       = cfg.get("cooldown_seconds", 30)
        self.margin_above     = cfg.get("margin_above_nose", 0.03)

        # Per-track state
        self._since = {}          # when gesture first detected
        self._last_detected = {}  # last timestamp gesture was detected
        self._cooldown = {}       # last trigger time
        self._last_option = {}    # last detected gesture option ("triangle_overhead" or "hands_overhead_v")

    def cleanup_track(self, tid: int) -> None:
        """Remove per-track state for a disappeared track."""
        self._since.pop(tid, None)
        self._last_detected.pop(tid, None)
        self._cooldown.pop(tid, None)
        self._last_option.pop(tid, None)

    def process(self, tid: int, kps, h: int, w: int, timestamp: float = None) -> dict:
        """Process one person's keypoints for hands-over-head / triangle gesture.

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
          is_confirmed     : bool — sustained for confirm_seconds (7-10s)
          time_held        : float — how long gesture has been held (seconds)
          triggered        : bool — confirmed AND not in cooldown
          gesture_option   : str — "triangle_overhead" | "hands_overhead_v" | None
          event_confidence : float — calculated event confidence (0.85 - 0.99)
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
            if now - self._last_detected.get(tid, -1e9) > 0.5:
                self._since.pop(tid, None)
                self._last_detected.pop(tid, None)
                self._last_option.pop(tid, None)
            return {
                "gesture_detected": False,
                "is_confirmed": False,
                "time_held": 0.0,
                "triggered": False,
                "gesture_option": None,
                "event_confidence": 0.0,
            }

        # Shoulders Y
        shoulder_y = float('inf')
        if _vis(l_shoulder, self.min_kp_conf) and _vis(r_shoulder, self.min_kp_conf):
            shoulder_y = min(float(l_shoulder[1]), float(r_shoulder[1]))

        # Calculate torso height if hips visible
        has_hips = _vis(l_hip, self.min_kp_conf) or _vis(r_hip, self.min_kp_conf)
        torso_h = 0.0
        avg_hip_y = 0.0
        if has_hips and shoulder_y != float('inf'):
            hips_y = [float(p[1]) for p in (l_hip, r_hip) if _vis(p, self.min_kp_conf)]
            avg_hip_y = sum(hips_y) / len(hips_y)
            torso_h = max(10.0, avg_hip_y - shoulder_y)

        # Determine head reference Y (use top of visible head keypoints 0..4, or estimate from torso)
        head_pts = [_kp(kps, i) for i in (0, 1, 2, 3, 4) if _vis(_kp(kps, i), self.min_kp_conf)]
        if head_pts:
            ref_y = min(float(p[1]) for p in head_pts)
        elif torso_h > 0.0:
            ref_y = shoulder_y - 0.25 * torso_h
        else:
            ref_y = shoulder_y - margin_px

        # Both wrists must be ABOVE shoulders, and near/above head level
        l_wrist_y = float(l_wrist[1])
        r_wrist_y = float(r_wrist[1])
        shoulder_w = abs(float(l_shoulder[0]) - float(r_shoulder[0]))
        wrist_sep = abs(float(l_wrist[0]) - float(r_wrist[0]))

        # Elbow separation for triangle shape check
        elbow_sep = 0.0
        if _vis(l_elbow, self.min_kp_conf) and _vis(r_elbow, self.min_kp_conf):
            elbow_sep = abs(float(l_elbow[0]) - float(r_elbow[0]))

        # ── Option 1: Triangle / Roof gesture above head ──
        # Both wrists above head, close together / meeting (apex of triangle),
        # with elbows spread outward (base of triangle).
        is_triangle = (
            l_wrist_y <= shoulder_y - 10.0 and
            r_wrist_y <= shoulder_y - 10.0 and
            l_wrist_y <= ref_y + 10.0 and
            r_wrist_y <= ref_y + 10.0 and
            wrist_sep <= max(35.0, 0.40 * shoulder_w) and
            elbow_sep >= max(wrist_sep * 1.35, 0.48 * shoulder_w)
        )

        # ── Option 2: Hands Raised V-Shape ──
        # Both wrists above shoulders and at/above head, spread apart.
        min_sep = max(18.0, 0.50 * shoulder_w)
        is_hands_up_v = (
            wrist_sep >= min_sep and
            l_wrist_y <= shoulder_y - 12.0 and
            r_wrist_y <= shoulder_y - 12.0 and
            l_wrist_y <= ref_y + 5.0 and
            r_wrist_y <= ref_y + 5.0
        )

        gesture_option = None
        if is_triangle:
            gesture_option = "triangle_overhead"
        elif is_hands_up_v:
            gesture_option = "hands_overhead_v"

        gesture_detected = (gesture_option is not None)

        # ── Check Upright Torso ──
        if has_hips and shoulder_y != float('inf'):
            if ref_y >= avg_hip_y - margin_px:
                gesture_detected = False
                gesture_option = None

        # ── Check Elbows ──
        # Wrists should be at or above elbows when elbows are visible
        if _vis(l_elbow, self.min_kp_conf) and l_wrist_y > float(l_elbow[1]) + margin_px:
            gesture_detected = False
            gesture_option = None
        if _vis(r_elbow, self.min_kp_conf) and r_wrist_y > float(r_elbow[1]) + margin_px:
            gesture_detected = False
            gesture_option = None

        # In hands over head, elbows should be above torso center (not resting down at waist/desk)
        if has_hips and shoulder_y != float('inf'):
            torso_mid = (shoulder_y + avg_hip_y) / 2.0
            if _vis(l_elbow, self.min_kp_conf) and float(l_elbow[1]) > torso_mid:
                gesture_detected = False
                gesture_option = None
            if _vis(r_elbow, self.min_kp_conf) and float(r_elbow[1]) > torso_mid:
                gesture_detected = False
                gesture_option = None

        # Temporal confirmation with 0.8s grace period to tolerate camera / tracker jitter
        if gesture_detected:
            if tid not in self._since or (now - self._last_detected.get(tid, -1e9)) > 0.8:
                self._since[tid] = now
            self._last_detected[tid] = now
            self._last_option[tid] = gesture_option
            time_held = now - self._since[tid]
        else:
            if tid in self._since and (now - self._last_detected.get(tid, -1e9)) <= 0.8:
                time_held = now - self._since[tid]
                gesture_option = self._last_option.get(tid, "hands_overhead_v")
            else:
                self._since.pop(tid, None)
                self._last_detected.pop(tid, None)
                self._last_option.pop(tid, None)
                time_held = 0.0

        is_confirmed = gesture_detected and time_held >= self.confirm_s

        # Calculate event confidence (0.85 - 0.99) based on landmark confidences and hold duration
        event_conf = 0.0
        if gesture_detected or time_held > 0:
            relevant_kps = [l_wrist, r_wrist, l_shoulder, r_shoulder]
            confs = [float(kp[2]) for kp in relevant_kps if _vis(kp, self.min_kp_conf)]
            avg_kp = sum(confs) / max(1, len(confs))
            duration_factor = min(1.0, time_held / max(1.0, self.confirm_s))
            event_conf = round(min(0.99, max(0.85, 0.80 + 0.12 * avg_kp + 0.07 * duration_factor)), 2)

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
            "gesture_option": gesture_option,
            "event_confidence": event_conf,
        }
