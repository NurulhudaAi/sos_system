import time
import logging
import cv2
import numpy as np
from collections import deque

logger = logging.getLogger("fall_detector")

L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW,    R_ELBOW    = 7, 8
L_WRIST,    R_WRIST    = 9, 10
L_HIP,      R_HIP      = 11, 12
L_KNEE,     R_KNEE     = 13, 14
L_ANKLE,    R_ANKLE    = 15, 16


def _kp(kps, i):
    return kps[i] if i < len(kps) else np.zeros(3)


def _vis(kp, c=0.3):
    return float(kp[2]) >= c


class FallDetector:
    """Fall detector with temporal confirmation, velocity/geometry signals, and trip suppression.

    Detection pipeline (per track ID, per frame):
      1. Geometry  — bbox ratio + spine angle detect horizontal posture
      2. Motion    — downward velocity spike OR sustained low motion
      3. is_down   — geometry AND (spike OR low_motion) OR strong_geometry_fallback
                     OR model_lying (NEW — custom laying/standing classifier, if provided)
      4. Ground    — sustained detect_ground → ground_time confirmed
      5. Confirm   — ground_time ≥ confirm_s OR geometry_fallen (3 s shortcut)
                     OR model_lying_fallen (NEW — model-confirm shortcut)
      6. Suppress  — recovered_quickly (nap/trip < recover_s) blocks is_fallen

    Config keys (all read from thresholds.yaml fall: section):
      bbox_ratio_thresh       1.3    bbox w/h > thresh → may be lying
      spine_angle_thresh      55.0   torso angle < thresh (°) → near horizontal
      warning_seconds         2.0    is_down_basic sustained → slow-fall fallback
      confirm_seconds         5.0    ground_time ≥ this → is_fallen 
      critical_seconds        30.0   time_lying ≥ this → is_critical
      history_len             16     frames kept in velocity deque
      spike_thresh_norm       0.18   downward vel / frame_height per second
      motion_thresh_norm      0.02   low-motion threshold (norm)
      sustain_seconds         1.0    seconds of ground signal before confirming
      lying_angle_thresh      40.0   torso angle for ground confirmation
      geometry_ratio_thresh   1.8    strong geometry ratio
      geometry_angle_thresh   25.0   strong geometry angle (°)
      geometry_confirm_seconds 3.0   strong geometry sustained → is_fallen shortcut
      danger_lying_seconds    15.0   lying ≥ this → danger_lying flag
      recover_seconds         5.0    lying < this then recovered → trip, no alert

      # ── NEW: last-known-angle memory (fixes "angle defaults to 90°
      #    when keypoints are missing" silently disabling angle path) ──
      angle_memory_seconds     2.0    reuse last CONFIDENT spine angle for
                                       up to this many seconds when pose
                                       keypoints drop below confidence
                                       (occlusion / prone / bad camera angle)

      # ── NEW: optional custom laying/standing classifier shortcut.
      #    Pairs with a fine-tuned pose model (see FallGuard notebook)
      #    that outputs a direct laying/standing class per detection,
      #    fed in via process(..., posture_class=, posture_conf=). This
      #    bypasses geometry/angle math entirely when the model is
      #    confident — useful because the generic COCO pose model's
      #    confidence (and keypoint visibility) often degrades for
      #    people lying flat, which otherwise starves the whole
      #    pipeline of a signal to work with. ──
      min_model_conf           0.5    min posture_conf to trust model signal
      model_confirm_seconds    2.0    model says "laying" sustained ≥ this → is_fallen shortcut
    """

    def __init__(self, cfg: dict):
        self.bbox_thresh           = cfg.get("bbox_ratio_thresh",      1.3)
        self.angle_thresh          = cfg.get("spine_angle_thresh",     55.0)
        self.warning_s             = cfg.get("warning_seconds",         2.0)  # FIX 2: slow-fall fallback
        self.confirm_s             = cfg.get("confirm_seconds",         4.0)  # FIX 1: was 8.0/9.0
        self.critical_s            = cfg.get("critical_seconds",       30.0)
        self.history_len           = cfg.get("history_len",            16)
        self.spike_thresh_norm     = cfg.get("spike_thresh_norm",       0.18)
        self.motion_thresh_norm    = cfg.get("motion_thresh_norm",      0.02)
        self.sustain_s             = cfg.get("sustain_seconds",         1.0)
        self.lying_angle_thresh    = cfg.get("lying_angle_thresh",     40.0)
        self.geometry_ratio_thresh = cfg.get("geometry_ratio_thresh",   1.8)
        self.geometry_angle_thresh = cfg.get("geometry_angle_thresh",  25.0)
        self.geometry_confirm_s    = cfg.get("geometry_confirm_seconds", 3.0)
        self.danger_lying_s        = cfg.get("danger_lying_seconds",   15.0)
        self.recover_s             = cfg.get("recover_seconds",         5.0)

        # NEW
        self.angle_memory_s        = cfg.get("angle_memory_seconds",    2.0)
        self.min_model_conf        = cfg.get("min_model_conf",          0.5)
        self.model_confirm_s       = cfg.get("model_confirm_seconds",   2.0)

        # --- per-track state dicts ---
        self._since            = {}   # when is_down first started
        self._hist             = {}   # deque of (timestamp, centroid_y)
        self._spike_time       = {}   # timestamp of first downward spike
        self._ground_time      = {}   # timestamp ground_contact confirmed
        self._height_hist      = {}   # deque of normed bbox heights
        self._stand_height     = {}   # FIX 3: rolling-max standing height (not frozen at 3 frames)
        self._ground_candidate = {}   # first detect_ground timestamp (waiting sustain)
        self._lying_geom_since = {}   # when strong geometry first persisted
        self._recovered_short  = {}   # timestamp of quick recovery (trip suppression)

        # Pre-existing static resting vs dynamic fall transition tracking
        self._was_upright           = {}   # tid -> bool (ever observed standing/upright)
        self._upright_streak        = {}   # tid -> int (consecutive upright frames)
        self._stand_ref             = {}   # tid -> (bbox_h_norm, box_w, y1, y2, ratio) standing reference
        self._initial_posture_lying = {}   # tid -> bool (appeared already horizontal on first frames)
        self._has_fall_transition   = {}   # tid -> bool (experienced dynamic fall transition / collapse)
        self._seated_since          = {}   # tid -> float timestamp when continuous sitting started
        self._seat_anchor           = {}   # tid -> (cx, cy, box_w, box_h) confirmed furniture seat anchor
        self._foreshortened_since   = {}   # tid -> float timestamp for foreshortened floor fall hysteresis
        self._recovery_streak       = {}   # tid -> int consecutive upright/sitting recovery frames

        # NEW: last confidently-measured spine angle per track, so a
        # frame with missing/low-confidence keypoints doesn't silently
        # fall back to "90° = standing" and disable the angle signal.
        self._last_angle       = {}   # tid -> angle (deg)
        self._last_angle_time  = {}   # tid -> timestamp of that measurement

        # NEW: custom laying/standing model shortcut bookkeeping
        self._model_lying_since = {}  # tid -> timestamp model first said "laying" (sustained)

    # ------------------------------------------------------------------
    # FIX 4: cleanup_track — call from main.py when tracker drops a tid.
    #         Prevents stale state from bleeding into reused track IDs.
    # ------------------------------------------------------------------
    def cleanup_track(self, tid: int) -> None:
        """Remove all per-track state for a disappeared track ID.

        Call this from main.py whenever the tracker drops `tid` from the scene,
        e.g. when the person leaves the frame or the tracker loses the target.
        Without this, a reused tid inherits the previous person's state, which
        can suppress a real fall (recovered_short still set) or fake one.
        """
        for d in (
            self._since,
            self._hist,
            self._spike_time,
            self._ground_time,
            self._height_hist,
            self._stand_height,
            self._ground_candidate,
            self._lying_geom_since,
            self._recovered_short,
            self._last_angle,
            self._last_angle_time,
            self._model_lying_since,
            self._was_upright,
            self._upright_streak,
            self._stand_ref,
            self._initial_posture_lying,
            self._has_fall_transition,
            self._seated_since,
            self._seat_anchor,
            self._foreshortened_since,
            self._recovery_streak,
        ):
            d.pop(tid, None)
        logger.debug("cleanup_track: cleared state for tid=%s", tid)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_hist(self, tid: int, now: float, cy: float) -> deque:
        dq = self._hist.get(tid)
        if dq is None:
            dq = deque(maxlen=self.history_len)
            self._hist[tid] = dq
        dq.append((now, float(cy)))
        return dq

    def _compute_velocities(self, dq):
        """Return (last_instantaneous_v, list_of_all_v) in pixels/sec."""
        if len(dq) < 2:
            return 0.0, []
        vels = []
        for i in range(1, len(dq)):
            t0, y0 = dq[i - 1]
            t1, y1 = dq[i]
            dt = max(1e-6, t1 - t0)
            vels.append((y1 - y0) / dt)
        return vels[-1], vels

    def _low_motion_in_window(self, dq, h_pixels: float) -> bool:
        """Return True if mean normalised |velocity| over last sustain_s is below threshold."""
        if len(dq) < 2:
            return False
        span = 0.0
        idx = len(dq) - 1
        abs_vels = []
        while idx > 0 and span < self.sustain_s:
            t0, y0 = dq[idx - 1]
            t1, y1 = dq[idx]
            span += t1 - t0
            abs_vels.append(abs((y1 - y0) / max(1e-6, t1 - t0)) / h_pixels)
            idx -= 1
        if not abs_vels:
            return False
        return float(np.mean(abs_vels)) < self.motion_thresh_norm

    # ------------------------------------------------------------------
    # Main processing entry point
    # ------------------------------------------------------------------

    def _is_sitting(self, kps, bbox, was_upright: bool = False) -> bool:
        """Return True if the person is sitting upright in a chair/sofa/desk/stairs/floor (not lying on the ground).

        Key characteristics of sitting:
          - Bounding box ratio W/H between 0.40 and 2.2 (accommodates narrow frontal chairs to wide angles)
          - Shoulders are clearly above hips: Y_hip - Y_shoulder >= 0.06 * box_h
          - Torso vector is upright/diagonal (torso_angle >= 42.0 deg, allowing natural forward leaning)
          - Head is at top / above shoulders (if visible)
          - Desk/table occlusions and floor sitting supported
        """
        x1, y1, x2, y2 = bbox
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        ratio = box_w / box_h
        if ratio < 0.40 or ratio >= 2.2:
            return False

        ls, rs = _kp(kps, L_SHOULDER), _kp(kps, R_SHOULDER)
        lh, rh = _kp(kps, L_HIP),      _kp(kps, R_HIP)
        lk, rk = _kp(kps, L_KNEE),     _kp(kps, R_KNEE)
        la, ra = _kp(kps, L_ANKLE),    _kp(kps, R_ANKLE)

        sh_pts = [p for p in (ls, rs) if _vis(p, 0.25)]
        hip_pts = [p for p in (lh, rh) if _vis(p, 0.25)]
        knee_ys = [p[1] for p in (lk, rk) if _vis(p, 0.25)]
        ank_ys = [p[1] for p in (la, ra) if _vis(p, 0.25)]
        head_pts = [_kp(kps, i) for i in (0, 1, 2, 3, 4) if _vis(_kp(kps, i), 0.10)]

        if sh_pts and hip_pts:
            avg_sh_y = sum(p[1] for p in sh_pts) / len(sh_pts)
            avg_hip_y = sum(p[1] for p in hip_pts) / len(hip_pts)
            avg_sh_x = sum(p[0] for p in sh_pts) / len(sh_pts)
            avg_hip_x = sum(p[0] for p in hip_pts) / len(hip_pts)
            torso_dx = abs(avg_hip_x - avg_sh_x)
            torso_dy = avg_hip_y - avg_sh_y
            torso_angle = abs(np.degrees(np.arctan2(torso_dy, torso_dx + 1e-6)))

            # ── Check 0a: Universal Chair / Bench / Desk Sitting (full body) ──
            # Seated on furniture: Shoulders clearly above hips, hips clearly above knees, knees clearly above ankles.
            # In a fall on the floor, hips and knees are at the same level (knee_y - hip_y < 0.05 * box_h).
            # Guard: If ratio < 0.65 and legs are vertically extended (hip-to-ankle >= 34% of box_h and
            # hip-to-knee >= 16% of box_h), the person is standing/walking upright, NOT sitting.
            if knee_ys and ank_ys:
                avg_knee_y = sum(knee_ys) / len(knee_ys)
                avg_ank_y = sum(ank_ys) / len(ank_ys)
                if ratio < 0.65 and (avg_ank_y - avg_hip_y) >= 0.34 * box_h and (avg_knee_y - avg_hip_y) >= 0.16 * box_h:
                    return False
                if (
                    torso_dy >= 0.06 * box_h
                    and (avg_knee_y - avg_hip_y) >= 0.04 * box_h
                    and (avg_ank_y - avg_knee_y) >= 0.02 * box_h
                    and torso_angle >= 42.0
                    and (ratio >= 0.65 or (avg_knee_y - avg_hip_y) <= 0.15 * box_h)
                ):
                    return True

            # ── Check 0b: Seated behind desk / table (ankles occluded, knees visible) ──
            if knee_ys and not ank_ys and ratio >= 0.65:
                avg_knee_y = sum(knee_ys) / len(knee_ys)
                if (
                    torso_dy >= 0.06 * box_h
                    and (avg_knee_y - avg_hip_y) >= 0.02 * box_h
                    and torso_angle >= 42.0
                ):
                    return True

            # ── Check 0c: Seated behind desk / table (both knees & ankles occluded, pre-existing seated track) ──
            if not knee_ys and not ank_ys and ratio >= 0.65 and not was_upright:
                if torso_dy >= 0.10 * box_h and torso_angle >= 42.0:
                    if head_pts:
                        avg_head_y = sum(p[1] for p in head_pts) / len(head_pts)
                        if avg_head_y <= avg_sh_y + 0.05 * box_h:
                            return True
                    else:
                        return True

            # ── Check 0d: Reclining on bench / sofa with feet on floor or active hands (e.g. using phone) ──
            if 0.75 <= ratio <= 1.65 and torso_dy >= 0.03 * box_h and torso_angle >= 12.0 and not was_upright:
                if knee_ys and ank_ys:
                    avg_knee_y = sum(knee_ys) / len(knee_ys)
                    avg_ank_y = sum(ank_ys) / len(ank_ys)
                    if avg_knee_y >= avg_hip_y and avg_ank_y > avg_knee_y and (avg_ank_y - avg_hip_y) >= 0.18 * box_h:
                        return True
                lw, rw = _kp(kps, L_WRIST), _kp(kps, R_WRIST)
                le, re = _kp(kps, L_ELBOW), _kp(kps, R_ELBOW)
                vis_wrists = [p for p in (lw, rw) if _vis(p, 0.25)]
                vis_elbows = [p for p in (le, re) if _vis(p, 0.25)]
                if vis_wrists and vis_elbows:
                    min_wy = min(p[1] for p in vis_wrists)
                    max_ey = max(p[1] for p in vis_elbows)
                    if min_wy <= max_ey + 0.04 * box_h and min_wy <= avg_hip_y:
                        return True

            # ── Check 1: Sitting on floor / carpet / low mat ──
            # Upper body spans vertically across bbox: head at top, shoulders middle, hips at base.
            if head_pts and torso_dy > 0 and ratio >= 0.65 and not was_upright:
                avg_head_y = sum(p[1] for p in head_pts) / len(head_pts)
                if avg_head_y <= avg_sh_y + 0.05 * box_h and avg_sh_y <= avg_hip_y:
                    head_to_hip = avg_hip_y - avg_head_y
                    if head_to_hip >= 0.25 * box_h and avg_hip_y >= y1 + 0.40 * box_h and torso_angle >= 42.0:
                        if not knee_ys or (sum(knee_ys) / len(knee_ys) <= avg_hip_y + 0.18 * box_h):
                            return True

        return False

    def _is_crouching_or_bending(self, kps, bbox, was_upright: bool = False) -> bool:
        """Detects active crouch / squatting / tie shoes / refrigerator reach where torso is upright/angled over bent legs."""
        x1, y1, x2, y2 = bbox
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        ratio = box_w / box_h
        # Crouching/squatting compresses body height, raising W/H ratio to >= 0.58 (standing is typically 0.35-0.55).
        # Lying flat on the floor is typically ratio >= 1.35 or torso angle < 38 deg.
        if ratio < 0.58 or ratio >= 1.35:
            return False

        ls, rs = _kp(kps, L_SHOULDER), _kp(kps, R_SHOULDER)
        lh, rh = _kp(kps, L_HIP),      _kp(kps, R_HIP)
        lk, rk = _kp(kps, L_KNEE),     _kp(kps, R_KNEE)
        la, ra = _kp(kps, L_ANKLE),    _kp(kps, R_ANKLE)

        sh_pts = [p for p in (ls, rs) if _vis(p, 0.25)]
        hip_pts = [p for p in (lh, rh) if _vis(p, 0.25)]
        knee_pts = [p for p in (lk, rk) if _vis(p, 0.25)]
        ank_pts = [p for p in (la, ra) if _vis(p, 0.25)]
        head_pts = [_kp(kps, i) for i in (0, 1, 2, 3, 4) if _vis(_kp(kps, i), 0.10)]

        if sh_pts and hip_pts:
            avg_sh_y = sum(p[1] for p in sh_pts) / len(sh_pts)
            avg_hip_y = sum(p[1] for p in hip_pts) / len(hip_pts)
            avg_sh_x = sum(p[0] for p in sh_pts) / len(sh_pts)
            avg_hip_x = sum(p[0] for p in hip_pts) / len(hip_pts)

            torso_dy = avg_hip_y - avg_sh_y
            torso_dx = abs(avg_hip_x - avg_sh_x)
            torso_angle = abs(np.degrees(np.arctan2(torso_dy, torso_dx + 1e-6)))

            # Forward waist-bend / shoe-tying crouch over vertical supporting legs:
            # Hips remain elevated above vertically stacked knees and ankles while torso folds forward/downward.
            if knee_pts and ank_pts:
                avg_knee_y = sum(p[1] for p in knee_pts) / len(knee_pts)
                avg_knee_x = sum(p[0] for p in knee_pts) / len(knee_pts)
                avg_ank_y = sum(p[1] for p in ank_pts) / len(ank_pts)
                avg_ank_x = sum(p[0] for p in ank_pts) / len(ank_pts)
                if (
                    (avg_knee_y - avg_hip_y) >= 0.22 * box_h
                    and (avg_ank_y - avg_hip_y) >= 0.38 * box_h
                    and (avg_ank_y - avg_knee_y) >= 0.08 * box_h
                    and abs(avg_knee_x - avg_hip_x) <= 0.25 * box_w
                    and abs(avg_ank_x - avg_hip_x) <= 0.30 * box_w
                ):
                    return True

            # Torso must be vertically oriented (shoulders above hips by at least 8% of box_h)
            # and torso angle >= 38.0 deg (fallen person on floor has angle <= 35 deg).
            if torso_dy < 0.08 * box_h or torso_angle < 38.0:
                return False

            # Head (if visible) must be near top of body, above or level with shoulders
            if head_pts:
                avg_head_y = sum(p[1] for p in head_pts) / len(head_pts)
                if avg_head_y > avg_sh_y + 0.10 * box_h:
                    return False

            # If knees are visible: hips are compressed down near knee level (deep squat/crouch)
            if knee_pts:
                avg_knee_y = sum(p[1] for p in knee_pts) / len(knee_pts)
                if avg_knee_y >= avg_sh_y and abs(avg_hip_y - avg_knee_y) <= 0.18 * box_h:
                    return True
            elif ratio >= 0.68 and not was_upright:
                # Knees occluded (e.g. behind cabinet/fridge/desk): require wider crouched ratio
                # and hips in lower half of bounding box with upright spine
                if head_pts and avg_hip_y >= y1 + 0.50 * box_h and torso_angle >= 42.0:
                    return True
                if ank_pts:
                    avg_ank_y = sum(p[1] for p in ank_pts) / len(ank_pts)
                    if (y2 - avg_ank_y) <= 0.25 * box_h and avg_hip_y >= y1 + 0.48 * box_h and torso_angle >= 42.0:
                        return True

        return False

    def _estimate_3d_posture(self, kps, bbox, stand_ref=None, h_pixels: float = 1080.0) -> bool:
        """Estimate 3D vertical leg extension and overhead arm kinematics.

        Returns True if the person has 3D vertically extended standing legs
        (shoulders -> hips -> knees -> ankles in vertical standing chain, even under
        wide-angle lens corner distortion) OR has arms actively raised overhead while standing.
        Accounts for 3D perspective foreshortening: if the person's height has collapsed
        by >=18% with a downward head drop at the same ground depth, their legs are
        stretched along the floor (Z-axis), not vertically standing (Y-axis).
        """
        x1, y1, x2, y2 = bbox
        box_w = max(1.0, float(x2 - x1))
        box_h = max(1.0, float(y2 - y1))
        if (box_w / box_h) >= 0.82:
            return False
        if stand_ref is not None:
            ref_h_norm, ref_w, ref_y1, ref_y2, ref_ratio = stand_ref
            ref_h_px = max(30.0, ref_h_norm * h_pixels)
            head_drop_px = float(y1) - ref_y1
            if box_h <= 0.84 * ref_h_px and head_drop_px >= 0.06 * ref_h_px:
                return False

        ls, rs = _kp(kps, L_SHOULDER), _kp(kps, R_SHOULDER)
        lh, rh = _kp(kps, L_HIP),      _kp(kps, R_HIP)
        lk, rk = _kp(kps, L_KNEE),     _kp(kps, R_KNEE)
        la, ra = _kp(kps, L_ANKLE),    _kp(kps, R_ANKLE)
        lw, rw = _kp(kps, L_WRIST),    _kp(kps, R_WRIST)

        sh_pts = [p for p in (ls, rs) if _vis(p, 0.25)]
        hip_pts = [p for p in (lh, rh) if _vis(p, 0.25)]
        knee_pts = [p for p in (lk, rk) if _vis(p, 0.25)]
        ank_pts = [p for p in (la, ra) if _vis(p, 0.25)]
        wrist_pts = [p for p in (lw, rw) if _vis(p, 0.25)]

        if not (sh_pts and hip_pts):
            return False

        avg_sh_y = sum(p[1] for p in sh_pts) / len(sh_pts)
        avg_hip_y = sum(p[1] for p in hip_pts) / len(hip_pts)
        torso_dy = avg_hip_y - avg_sh_y
        if torso_dy < 0.10 * box_h:
            return False

        # 1. Full 3D vertical leg extension (hips -> knees -> ankles extended downward)
        if knee_pts and ank_pts:
            avg_knee_y = sum(p[1] for p in knee_pts) / len(knee_pts)
            avg_ank_y = sum(p[1] for p in ank_pts) / len(ank_pts)
            if (
                (avg_knee_y - avg_hip_y) >= 0.12 * box_h
                and (avg_ank_y - avg_hip_y) >= 0.28 * box_h
                and (avg_ank_y - avg_knee_y) >= 0.08 * box_h
            ):
                return True

        # 2. Upright torso with arms actively raised overhead (e.g. Hands-Up SOS gesture)
        if wrist_pts and torso_dy >= 0.12 * box_h:
            min_wy = min(p[1] for p in wrist_pts)
            if min_wy < avg_sh_y - 0.02 * box_h:
                return True

        return False

    def process(self, tid: int, kps, bbox, h: int, w: int,
                posture_class=None, posture_conf: float = 0.0,
                timestamp: float = None, on_furniture: bool = False) -> dict:
        """Process one detection for track `tid` and return a result dict.

        Parameters
        ----------
        tid  : track ID (int)
        kps  : keypoints array, shape (N, 3) — [x, y, confidence]
        bbox : [x1, y1, x2, y2] in pixels
        h, w : frame height and width in pixels
        posture_class : optional str/int — "laying"/"standing" or 0/1, from
                        an optional custom laying/standing classifier model
                        (e.g. the FallGuard fine-tuned pose model). Pass
                        None if not available — everything still works via
                        geometry/motion as before.
        posture_conf  : confidence for posture_class (0.0 if not available)
        timestamp     : optional float (video seconds for evaluation / playback;
                        defaults to time.time() for live stream)
        on_furniture  : True if the person's hip anchor lies inside a configured
                        furniture zone (bench/bed/sofa). Lying there is treated
                        as resting: fall timers are reset and no fall is emitted.
                        Leaving the furniture (e.g. falling off it) restarts
                        normal floor-based detection.
        """
        now = timestamp if timestamp is not None else time.time()
        x1, y1, x2, y2 = bbox
        ratio_raw = (x2 - x1) / ((y2 - y1) + 1e-6)
        # [Day 3] 4-Side Edge & Bottom Border Truncation Guard
        edge_px = 0.02 * float(w or 1)
        edge_py = 0.035 * float(h or 1)
        at_edge = (
            (x1 <= edge_px)
            or (x2 >= float(w or 1) - edge_px)
            or (y1 <= edge_py)
            or (y2 >= float(h or 1) - edge_py)
        )
        bottom_truncated = (
            (y2 >= float(h or 1) - edge_py)
            or (y1 >= 0.72 * float(h or 1))
        )
        ratio = min(ratio_raw, 1.0) if at_edge else ratio_raw

        # ── Spine angle (shoulder-midpoint → hip-midpoint) ──
        ls, rs = _kp(kps, L_SHOULDER), _kp(kps, R_SHOULDER)
        lh, rh = _kp(kps, L_HIP),     _kp(kps, R_HIP)
        has_pose = _vis(ls) and _vis(rs) and _vis(lh) and _vis(rh)

        torso_dy = 0.0
        if has_pose:
            sh = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2])
            hp = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])
            d  = sh - hp
            torso_dy = hp[1] - sh[1]  # positive when shoulders are clearly above hips
            angle = abs(np.degrees(np.arctan2(abs(d[1]), abs(d[0]) + 1e-6)))
            self._last_angle[tid] = angle
            self._last_angle_time[tid] = now
        else:
            last_a    = self._last_angle.get(tid)
            last_a_ts = self._last_angle_time.get(tid)
            if last_a is not None and last_a_ts is not None and (
                (now - last_a_ts) <= self.angle_memory_s
                or (last_a < 45.0 and ratio >= 1.10)
                or (tid in self._foreshortened_since)
                or (tid in self._ground_time)
            ):
                angle = last_a  # reuse recent confident reading (or persist while fallen/foreshortened)
            else:
                angle = 90.0     # no recent info at all — neutral default

        cx  = (x1 + x2) / 2.0
        cy  = (y1 + y2) / 2.0
        h_pixels = float(h) if h else 1.0

        bbox_h      = y2 - y1
        box_h       = max(1.0, float(bbox_h))
        box_w       = max(1.0, float(x2 - x1))
        bbox_h_norm = bbox_h / h_pixels if h_pixels else 0.0

        hh = self._height_hist.get(tid)
        if hh is None:
            hh = deque(maxlen=self.history_len)
            self._height_hist[tid] = hh
        hh.append(bbox_h_norm)

        # ── Posture classification checks (3D & raw geometry) ──
        was_upright_prev = self._was_upright.get(tid, False)
        is_sitting_geom = bool(self._is_sitting(kps, bbox, was_upright=was_upright_prev))
        is_crouching_geom = bool(self._is_crouching_or_bending(kps, bbox, was_upright=was_upright_prev))
        is_standing_legs_extended = bool(
            self._estimate_3d_posture(kps, bbox, stand_ref=self._stand_ref.get(tid), h_pixels=h_pixels)
        )

        # ── Seated-Furniture Anchor Memory (suppresses bench napping / reclining) ──
        is_seated_resting = False
        if is_sitting_geom and not was_upright_prev:
            self._seated_since.setdefault(tid, now)
            if (now - self._seated_since[tid]) >= 1.5:
                self._seat_anchor.setdefault(tid, (cx, cy, box_w, box_h))
        if tid in self._seat_anchor:
            acx, acy, aw, ah = self._seat_anchor[tid]
            if abs(cx - acx) <= 0.95 * max(box_w, aw) and cy <= acy + 0.25 * max(box_h, ah):
                is_seated_resting = True
                is_sitting_geom = True
            elif abs(cx - acx) > 1.30 * max(box_w, aw) or abs(cy - acy) > 0.50 * max(box_h, ah):
                self._seat_anchor.pop(tid, None)
                self._seated_since.pop(tid, None)

        # Genuine upright standing posture (distance-invariant):
        is_upright_now = bool(
            has_pose and (angle >= 50.0) and (torso_dy >= 0.10 * box_h) and (ratio <= 0.68)
            and not is_sitting_geom and not is_crouching_geom and not is_seated_resting
        )
        is_vert_bbox_now = bool(
            ratio <= 0.62 and box_h >= 40.0
            and not is_sitting_geom and not is_crouching_geom and not is_seated_resting
            and not at_edge and not bottom_truncated
        )

        # Track initial appearance posture
        if tid not in self._initial_posture_lying:
            is_init_lying = bool(ratio >= 0.80 or (has_pose and angle <= 38.0) or is_sitting_geom)
            self._initial_posture_lying[tid] = is_init_lying
            self._was_upright[tid] = False
            self._upright_streak[tid] = 0
            self._has_fall_transition[tid] = False
        elif not self._was_upright.get(tid, False) and (ratio >= 0.90 or (has_pose and angle <= 35.0) or is_seated_resting):
            self._initial_posture_lying[tid] = True
            self._upright_streak[tid] = 0

        # ── Standing calibration ──
        is_standing_frame = bool(
            not is_sitting_geom and not is_crouching_geom and not is_seated_resting
            and not at_edge and not bottom_truncated
            and (
                (has_pose and is_upright_now and angle >= 55.0 and torso_dy >= 0.12 * box_h and ratio <= 0.60)
                or (ratio <= 0.58 and box_h >= 50.0 and (not has_pose or angle >= 50.0))
            )
        )
        is_strict_standing = bool(
            has_pose and is_standing_frame and angle >= 65.0 and torso_dy >= 0.18 * box_h and ratio <= 0.55
        )
        if self._initial_posture_lying.get(tid, False):
            if is_strict_standing:
                streak = self._upright_streak.get(tid, 0) + 1
                self._upright_streak[tid] = streak
                if streak >= 6:
                    self._was_upright[tid] = True
                    self._initial_posture_lying[tid] = False
                    self._stand_ref[tid] = (bbox_h_norm, box_w, float(y1), float(y2), ratio)
                    prev = self._stand_height.get(tid, 0.0)
                    if bbox_h_norm > prev:
                        self._stand_height[tid] = bbox_h_norm
            else:
                self._upright_streak[tid] = 0
        elif is_standing_frame:
            streak = self._upright_streak.get(tid, 0) + 1
            self._upright_streak[tid] = streak
            if streak >= 3:
                self._was_upright[tid] = True
                prev = self._stand_height.get(tid, 0.0)
                if bbox_h_norm > prev:
                    self._stand_height[tid] = bbox_h_norm
                    prev = bbox_h_norm
                if tid not in self._stand_ref:
                    self._stand_ref[tid] = (bbox_h_norm, box_w, float(y1), float(y2), ratio)
                elif bbox_h_norm >= 0.88 * prev:
                    sref_h, sref_w, sref_y1, sref_y2, sref_r = self._stand_ref[tid]
                    self._stand_ref[tid] = (
                        max(sref_h, bbox_h_norm),
                        min(sref_w, box_w) if ratio <= 0.55 else sref_w,
                        min(sref_y1, float(y1)),
                        float(y2),
                        min(sref_r, ratio),
                    )
        elif (has_pose and angle < 45.0) or ratio >= 0.68 or is_sitting_geom:
            self._upright_streak[tid] = 0

        stand_h = self._stand_height.get(tid)
        stand_ref = self._stand_ref.get(tid)
        was_down = bool(tid in self._ground_time or tid in self._since or tid in self._foreshortened_since)
        thresh_factor = 0.88 if was_down else 0.85

        # Foreshortened floor collapse (person falls along the camera's line-of-sight, e.g. in an aisle):
        # Even if 2D projected spine angle is 40-70 deg, the person's head (y1) drops sharply toward
        # their feet (y2 stays on the floor), height shrinks relative to standing height, and aspect ratio widens.
        detect_foreshortened_fall = False
        if stand_ref is not None and self._was_upright.get(tid, False):
            ref_h_norm, ref_w, ref_y1, ref_y2, ref_ratio = stand_ref
            peak_h_norm = max(stand_h or 0.0, ref_h_norm)
            ref_h_px = max(30.0, peak_h_norm * h_pixels)
            head_drop_px = float(y1) - ref_y1
            feet_shift_px = float(y2) - ref_y2
            ref_cy = (ref_y1 + ref_y2) * 0.5

            lk, rk = _kp(kps, L_KNEE), _kp(kps, R_KNEE)
            la, ra = _kp(kps, L_ANKLE), _kp(kps, R_ANKLE)
            hip_pts = [p for p in (lh, rh) if _vis(p, 0.25)]
            knee_pts = [p for p in (lk, rk) if _vis(p, 0.25)]
            ank_pts = [p for p in (la, ra) if _vis(p, 0.25)]
            is_vertical_leg_support = False
            if hip_pts and knee_pts and ank_pts:
                avg_hip_y = sum(p[1] for p in hip_pts) / len(hip_pts)
                avg_hip_x = sum(p[0] for p in hip_pts) / len(hip_pts)
                avg_knee_y = sum(p[1] for p in knee_pts) / len(knee_pts)
                avg_knee_x = sum(p[0] for p in knee_pts) / len(knee_pts)
                avg_ank_y = sum(p[1] for p in ank_pts) / len(ank_pts)
                avg_ank_x = sum(p[0] for p in ank_pts) / len(ank_pts)
                if (
                    (avg_knee_y - avg_hip_y) >= 0.22 * box_h
                    and (avg_ank_y - avg_hip_y) >= 0.38 * box_h
                    and (avg_ank_y - avg_knee_y) >= 0.08 * box_h
                    and abs(avg_knee_x - avg_hip_x) <= 0.25 * box_w
                    and abs(avg_ank_x - avg_hip_x) <= 0.30 * box_w
                ):
                    is_vertical_leg_support = True

            guards_ok = bool(
                not is_seated_resting
                and not at_edge
                and not bottom_truncated
                and not is_standing_legs_extended
                and not is_vertical_leg_support
                and (not has_pose or torso_dy >= 0.08 * box_h)
            )
            entry_ok = bool(
                guards_ok
                and bbox_h_norm <= peak_h_norm * 0.85
                and head_drop_px >= 0.10 * ref_h_px
                and feet_shift_px >= -0.18 * ref_h_px
                and cy >= (ref_cy - 0.05 * ref_h_px)
                and 0.66 <= ratio <= 1.40
                and (ratio >= ref_ratio * 1.18 or box_w >= ref_w * 1.20)
                and (not has_pose or angle <= 70.5)
            )
            hold_ok = bool(
                (tid in self._foreshortened_since)
                and guards_ok
                and bbox_h_norm <= peak_h_norm * 0.90
                and head_drop_px >= 0.06 * ref_h_px
                and ratio >= 0.60
                and (not has_pose or angle <= 72.0)
            )
            if entry_ok or hold_ok:
                self._foreshortened_since.setdefault(tid, now)
                detect_foreshortened_fall = True
            else:
                self._foreshortened_since.pop(tid, None)
        else:
            self._foreshortened_since.pop(tid, None)

        # Stature collapse: height collapsed by >= 15% (or 12% if already down) compared to KNOWN standing height
        # ONLY valid if the person was actually calibrated standing (stand_h is not None and _was_upright is True)
        # AND either their torso tilted horizontally OR they suffered a foreshortened floor collapse in place.
        detect_lying_height = bool(
            (
                stand_h is not None and self._was_upright.get(tid, False)
                and (bbox_h_norm <= stand_h * thresh_factor)
                and (angle < 38.0 or (ratio >= 1.25 and angle < 45.0))
                and not is_crouching_geom
                and not is_sitting_geom
                and not is_seated_resting
            )
            or detect_foreshortened_fall
        )

        is_sitting = bool((is_sitting_geom or is_seated_resting) and not detect_lying_height)
        is_crouching = bool(is_crouching_geom and not detect_lying_height)
        is_upright = bool((is_upright_now or is_standing_legs_extended) and not detect_lying_height)
        is_vert_bbox = bool(is_vert_bbox_now and not detect_lying_height)

        # ── Velocity history ──
        dq = self._update_hist(tid, now, cy)
        last_v, vels = self._compute_velocities(dq)
        avg_abs_v      = float(np.mean(np.abs(vels))) if vels else 0.0
        last_v_norm    = last_v    / h_pixels
        avg_abs_v_norm = avg_abs_v / h_pixels

        # Require genuine downward centroid drop (at least 22% of reference body height)
        # AND that the person was not already lying down prior to the spike,
        # so bbox expansion on a stationary horizontal bench sleeper cannot fake a velocity spike.
        min_cy_window = min((pt_y for _, pt_y in dq), default=cy)
        ref_h_px = max(box_h, (stand_h * h_pixels) if stand_h is not None else box_h)
        net_drop_px = cy - min_cy_window
        last_v_body = last_v / max(30.0, ref_h_px)
        already_lying_long = bool(
            (tid in self._ground_time and (now - self._ground_time[tid]) > 2.0)
            or (self._initial_posture_lying.get(tid, False) and not self._was_upright.get(tid, False))
        )
        detect_spike = bool(
            not already_lying_long
            and (last_v_norm > self.spike_thresh_norm or (self._was_upright.get(tid, False) and last_v_body > 0.75))
            and (net_drop_px >= 0.22 * ref_h_px)
        )
        if detect_spike or detect_lying_height:
            self._has_fall_transition[tid] = True

        if self._was_upright.get(tid, False) and (angle <= self.lying_angle_thresh or ratio >= self.bbox_thresh):
            self._has_fall_transition[tid] = True

        # ════════════════════════════════════════════════════════════════
        # STAGE 1 — Geometry signals
        # ════════════════════════════════════════════════════════════════
        if is_upright or is_sitting or is_crouching:
            is_down_basic = False
        elif detect_lying_height:
            is_down_basic = True
        elif has_pose:
            # Torso is tilted horizontally: angle < 38.0 OR strongly horizontal bbox with non-vertical spine
            is_down_basic = (angle < 38.0) or (ratio >= self.geometry_ratio_thresh and angle < 45.0)
        else:
            is_down_basic = ratio > self.bbox_thresh or angle < 38.0

        # Strong geometry: very wide bbox AND very flat torso simultaneously.
        strong_lying_geometry = bool(
            not is_upright and
            ratio >= self.geometry_ratio_thresh and
            angle <= self.geometry_angle_thresh
        )

        if strong_lying_geometry:
            self._lying_geom_since.setdefault(tid, now)
        else:
            self._lying_geom_since.pop(tid, None)

        geom_start     = self._lying_geom_since.get(tid)
        geometry_time  = (now - geom_start) if geom_start else 0.0
        geometry_down   = strong_lying_geometry and geometry_time >= self.sustain_s
        geometry_fallen = strong_lying_geometry and geometry_time >= self.geometry_confirm_s

        # ════════════════════════════════════════════════════════════════
        # STAGE 1b — NEW: custom laying/standing model signal
        #
        # If an optional fine-tuned classifier (posture_class/posture_conf)
        # is supplied, treat a confident "laying" call as a strong,
        # geometry-independent signal — this is the whole point of
        # training such a model: it doesn't need clean keypoints to work,
        # so it catches cases where pose estimation itself is degraded
        # (prone posture, occlusion, odd camera angle) and the geometry
        # path above has nothing reliable to go on.
        # ════════════════════════════════════════════════════════════════
        is_laying_label = str(posture_class).lower() in ("laying", "lying", "0")
        model_confident_lying = bool(
            posture_class is not None and is_laying_label and posture_conf >= self.min_model_conf
        )
        if model_confident_lying:
            self._model_lying_since.setdefault(tid, now)
        else:
            self._model_lying_since.pop(tid, None)

        model_lying_start  = self._model_lying_since.get(tid)
        model_lying_time   = (now - model_lying_start) if model_lying_start else 0.0
        model_lying_fallen = model_confident_lying and model_lying_time >= self.model_confirm_s

        # ════════════════════════════════════════════════════════════════
        # STAGE 2 — Motion signals
        # ════════════════════════════════════════════════════════════════
        low_motion   = self._low_motion_in_window(dq, h_pixels)

        # ════════════════════════════════════════════════════════════════
        # STAGE 3 — is_down decision
        #
        # FIX 2: Slow medical collapse (no spike, still moving slightly) was
        # a false-negative because it never cleared (spike OR low_motion).
        # Added a time-based fallback: if is_down_basic has been true for
        # warning_seconds, accept is_down regardless of velocity signals.
        # This catches gradual slumps that don't produce a velocity spike
        # and haven't stilled yet.
        # ════════════════════════════════════════════════════════════════
        since_down = self._since.get(tid)
        slow_fall_fallback = (
            is_down_basic
            and not is_sitting
            and not is_crouching
            and since_down is not None
            and (now - since_down) >= self.warning_s
        )

        # ── Posture classification checks ──
        is_sitting = bool((is_sitting_geom or is_seated_resting) and not detect_lying_height)
        is_crouching = bool(is_crouching_geom and not detect_lying_height)

        if is_upright or is_sitting:
            is_down = False
            rec_cnt = self._recovery_streak.get(tid, 0) + 1
            self._recovery_streak[tid] = rec_cnt
            if tid not in self._ground_time or rec_cnt >= 2 or is_seated_resting:
                self._since.pop(tid, None)
                self._ground_candidate.pop(tid, None)
                self._ground_time.pop(tid, None)
                self._lying_geom_since.pop(tid, None)
                self._spike_time.pop(tid, None)
        elif is_crouching and not detect_spike:
            # Person is crouching/bending on their feet without an impact velocity spike
            is_down = False
            rec_cnt = self._recovery_streak.get(tid, 0) + 1
            self._recovery_streak[tid] = rec_cnt
            if tid not in self._ground_time or rec_cnt >= 2:
                self._since.pop(tid, None)
                self._ground_candidate.pop(tid, None)
                self._ground_time.pop(tid, None)
                self._lying_geom_since.pop(tid, None)
                self._spike_time.pop(tid, None)
        else:
            self._recovery_streak[tid] = 0
            is_down = (
                (is_down_basic and (detect_spike or low_motion))
                or geometry_down
                or slow_fall_fallback   # FIX 2
                or model_confident_lying  # NEW: model says laying → count as down
                or detect_lying_height
            )

        # ── Debug logging: per-track signal breakdown ──
        if is_down:
            logger.debug(
                "tid=%s IS_DOWN: ratio=%.2f angle=%.1f has_pose=%s "
                "spike=%s low_motion=%s geom_down=%s geom_time=%.1f "
                "slow_fb=%s model_lying=%s model_time=%.1f "
                "stale_angle=%s posture=%s posture_conf=%.2f",
                tid, ratio, angle, has_pose,
                detect_spike, low_motion, geometry_down, geometry_time,
                slow_fall_fallback, model_confident_lying, model_lying_time,
                (not has_pose and angle != 90.0),
                posture_class, posture_conf,
            )

        # ── Spike bookkeeping ──
        if detect_spike:
            self._spike_time.setdefault(tid, now)
        else:
            # Hold spike marker across confirm/dispatch window to measure time_to_ground accurately
            if tid in self._spike_time and now - self._spike_time[tid] > max(12.0, self.confirm_s * 3.0):
                self._spike_time.pop(tid, None)

        # ════════════════════════════════════════════════════════════════
        # STAGE 5 — Ground contact detection
        # ════════════════════════════════════════════════════════════════
        detect_ground = (
            (low_motion and (ratio > self.bbox_thresh or angle < self.lying_angle_thresh))
            or detect_lying_height
            or geometry_down
            or model_confident_lying   # NEW
        )

        # ── Upright / Sitting / Crouch Posture Guard ──
        # Genuine upright torso: vertical spine, substantial vertical torso (distance-invariant)
        has_upright_torso = bool(
            has_pose and (angle >= 48.0) and (torso_dy >= 0.08 * box_h)
            and not detect_lying_height
        )

        if is_sitting or has_upright_torso or is_upright or (is_crouching and not detect_spike):
            is_down = False
            detect_ground = False
            rec_cnt = self._recovery_streak.get(tid, 0)
            if tid not in self._ground_time or rec_cnt >= 2 or is_seated_resting:
                self._ground_candidate.pop(tid, None)
                self._ground_time.pop(tid, None)
                self._since.pop(tid, None)
                self._lying_geom_since.pop(tid, None)
                self._spike_time.pop(tid, None)

        if detect_ground:
            self._ground_candidate.setdefault(tid, now)
            if (
                tid not in self._ground_time
                and (now - self._ground_candidate[tid]) >= self.sustain_s
            ):
                self._ground_time[tid] = self._ground_candidate[tid]
        else:
            self._ground_candidate.pop(tid, None)
            # Person left the ground — check for quick recovery (trip/stumble)
            if tid in self._ground_time and not is_down and (self._recovery_streak.get(tid, 0) >= 2 or is_seated_resting):
                ground_start = self._ground_time.pop(tid, None)
                if ground_start is not None:
                    lying_d = now - ground_start
                    if lying_d < self.recover_s:
                        self._recovered_short[tid] = now
                        logger.debug(
                            "tid=%s quick recovery (%.1fs < %.1fs) → suppressed",
                            tid, lying_d, self.recover_s,
                        )

        # ── is_down / since bookkeeping ──
        if is_down:
            self._since.setdefault(tid, now)
            td = now - self._since[tid]
        else:
            if tid not in self._ground_time or self._recovery_streak.get(tid, 0) >= 2 or is_seated_resting:
                self._since.pop(tid, None)
                td = 0.0
                self._spike_time.pop(tid, None)
                if is_upright or is_sitting or has_upright_torso:
                    self._ground_time.pop(tid, None)
            else:
                td = (now - self._since[tid]) if tid in self._since else 0.0

        # ── Geometry path can set ground_time independently ──
        ground_t = self._ground_time.get(tid)
        if geometry_fallen and geom_start is not None and ground_t is None:
            self._ground_time[tid] = geom_start
            ground_t = geom_start

        # ── NEW: model path can set ground_time independently too ──
        if model_lying_fallen and model_lying_start is not None and ground_t is None:
            self._ground_time[tid] = model_lying_start
            ground_t = model_lying_start

        # ════════════════════════════════════════════════════════════════
        # STAGE 6 — Timing derivations
        # ════════════════════════════════════════════════════════════════
        spike_t        = self._spike_time.get(tid)
        time_to_ground = None
        time_lying     = None
        danger_lying   = False

        if spike_t and ground_t:
            time_to_ground = round(max(0.0, ground_t - spike_t), 2)
        if ground_t:
            time_lying = round(max(0.0, now - ground_t), 1)
            if time_lying >= self.danger_lying_s:
                danger_lying = True

        # ════════════════════════════════════════════════════════════════
        # STAGE 7 — Fall confirmation
        #
        # FIX 1: confirm_seconds default changed from 8–9 s → 5 s.
        #         The yaml value overrides this, but the in-code default
        #         is now correct.  A confirmed fall requires either:
        #           (a) ground contact sustained ≥ confirm_s seconds, OR
        #           (b) geometry_fallen (strong posture ≥ geometry_confirm_s), OR
        #           (c) model_lying_fallen (NEW — custom classifier ≥ model_confirm_s)
        #         Quick recoveries (trip/stumble) suppress the flag.
        # ════════════════════════════════════════════════════════════════
        if ground_t:
            is_fallen = (now - ground_t) >= self.confirm_s
        else:
            is_fallen = False

        if geometry_fallen:
            is_fallen = True  # geometry shortcut (3 s of clear lying posture)
            logger.info(
                "tid=%s GEOMETRY SHORTCUT fired: geom_time=%.1f ratio=%.2f angle=%.1f",
                tid, geometry_time, ratio, angle,
            )

        if model_lying_fallen:
            is_fallen = True  # NEW: model shortcut (2 s of confident "laying" classification)
            logger.info(
                "tid=%s MODEL SHORTCUT fired: model_time=%.1f posture=%s conf=%.2f",
                tid, model_lying_time, posture_class, posture_conf,
            )

        # Sitting, upright, or crouching person is never fallen
        if is_sitting or has_upright_torso or is_upright or (is_crouching and not detect_spike):
            is_fallen = False
            danger_lying = False
            is_critical = False

        # ── Pre-existing resting/lying on furniture suppression ──
        # A true Fall requires evidence of a fall event:
        # Either the person was observed standing/upright prior to being down,
        # or they experienced a downward velocity spike / impact,
        # or a dynamic stature collapse transition from a calibrated standing height.
        # A static/resting person on a bench or chair who was never seen standing
        # and had no velocity spike did NOT suffer a fall event.
        has_fall_evidence = bool(
            self._was_upright.get(tid, False)
            or self._has_fall_transition.get(tid, False)
            or detect_spike
            or (spike_t is not None)
            or (self._was_upright.get(tid, False) and detect_lying_height)
        )
        if not has_fall_evidence:
            is_down = False
            is_fallen = False
            danger_lying = False
            is_critical = False
            ground_t = None
            time_lying = None
            self._since.pop(tid, None)
            self._ground_candidate.pop(tid, None)
            self._ground_time.pop(tid, None)
            self._lying_geom_since.pop(tid, None)

        # ── Trip / quick-recovery suppression ──
        recovered_quickly = False
        if tid in self._recovered_short:
            tmark = self._recovered_short[tid]
            if (now - tmark) < (self.recover_s * 2):
                recovered_quickly = True
            else:
                self._recovered_short.pop(tid, None)

        if recovered_quickly:
            is_fallen = False

        # ── Time-down value ──
        time_down_val = (now - ground_t) if ground_t else td

        # ── Critical escalation ──
        is_critical = bool(
            has_fall_evidence and not is_sitting and not is_upright and not has_upright_torso and (
                (ground_t is not None and time_lying is not None and time_lying >= self.critical_s)
                or danger_lying
            )
        )

        # ── Furniture guard (bench / bed / sofa zone) ──
        # Lying on furniture is resting, not a floor fall. Reset all down/ground
        # timers so that if the person later ends up on the floor (e.g. falls off
        # the bench) the 4s confirm window starts fresh from the floor.
        if on_furniture:
            is_down = False
            is_fallen = False
            danger_lying = False
            is_critical = False
            ground_t = None
            time_lying = None
            time_down_val = 0.0
            self._since.pop(tid, None)
            self._ground_candidate.pop(tid, None)
            self._ground_time.pop(tid, None)
            self._lying_geom_since.pop(tid, None)
            self._model_lying_since.pop(tid, None)
            self._foreshortened_since.pop(tid, None)

        # ── Event Confidence (Fall Certainty) ──
        event_conf = 0.0
        if is_fallen or is_down:
            angle_cert = max(0.0, min(1.0, (90.0 - angle) / 65.0))
            ratio_cert = max(0.0, min(1.0, (ratio - 1.0) / 0.8)) if ratio else 0.5
            spike_cert = 1.0 if spike_t else 0.0
            model_cert = posture_conf if (posture_class == "laying" and posture_conf > 0.3) else 0.75
            lying_time_cert = min(1.0, (time_lying or 0.0) / 6.0)

            raw_event_conf = 0.72 + 0.10 * angle_cert + 0.06 * ratio_cert + 0.04 * spike_cert + 0.04 * lying_time_cert + 0.04 * model_cert
            event_conf = round(min(0.99, max(0.80, raw_event_conf)), 2)

        return {
            "is_fallen":                 bool(is_fallen),
            "is_down":                   bool(is_down),
            "has_fall_evidence":         bool(has_fall_evidence),
            "on_furniture":              bool(on_furniture),
            "is_sitting":                bool(is_sitting),
            "is_seated_resting":         bool(is_seated_resting),
            "is_crouching":              bool(is_crouching),
            "detect_foreshortened_fall": bool(detect_foreshortened_fall),
            "time_down":                 round(time_down_val, 1),
            "bbox_ratio":                round(ratio, 2),
            "spine_angle":               round(angle, 1),
            "vel_y":                     round(last_v, 1),
            "avg_vel":                   round(avg_abs_v, 1),
            "vel_y_norm":                round(last_v_norm, 4),
            "avg_vel_norm":              round(avg_abs_v_norm, 4),
            "is_critical":               bool(is_critical),
            "event_confidence":          event_conf,
            "spike_time":                spike_t,
            "ground_time":               ground_t,
            "time_to_ground":            time_to_ground,
            "time_lying":                time_lying,
            "danger_lying":              bool(danger_lying),
            "recovered_quickly":         bool(recovered_quickly),
            "strong_lying_geometry":     bool(strong_lying_geometry),
            "geometry_time":             round(geometry_time, 1),
            "slow_fall_fallback":        bool(slow_fall_fallback),   # debug signal
            "used_stale_angle":          bool(not has_pose and angle != 90.0),  # NEW debug signal
            "model_confident_lying":     bool(model_confident_lying),           # NEW debug signal
            "model_lying_time":          round(model_lying_time, 1),            # NEW debug signal
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def draw(self, frame, tid: int, res: dict, bbox, kps, conf: float = 0.0):
        x1, y1, x2, y2 = [int(v) for v in bbox]

        if res.get("is_critical"):
            c = (0, 0, 220)
        elif res.get("is_fallen"):
            c = (0, 0, 200)
        elif res.get("is_down"):
            c = (0, 140, 255)
        else:
            c = (0, 200, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        cv2.putText(
            frame, f"ID:{tid} {conf:.0%}",
            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1,
        )

        if res["is_down"] and res["time_down"] > 0:
            label = f"DOWN {res['time_down']}s"
            if res.get("slow_fall_fallback"):
                label += " [slow]"
            if res.get("used_stale_angle"):
                label += " [mem]"
            if res.get("model_confident_lying"):
                label += " [model]"
            cv2.putText(
                frame, label,
                (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2,
            )

        if "vel_y" in res:
            cv2.putText(
                frame,
                f"vel:{res['vel_y']}px/s avg:{res.get('avg_vel', 0)} geom:{res.get('geometry_time', 0)}s",
                (x1, y2 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
            )

        if res.get("is_critical"):
            cv2.putText(
                frame, "CRITICAL",
                (x1, y2 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2,
            )

        # Skeleton
        for a, b in [
            (L_SHOULDER, R_SHOULDER),
            (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
            (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
            (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
            (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
            (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
        ]:
            ka, kb = _kp(kps, a), _kp(kps, b)
            if _vis(ka) and _vis(kb):
                cv2.line(
                    frame,
                    (int(ka[0]), int(ka[1])),
                    (int(kb[0]), int(kb[1])),
                    (160, 160, 160), 1,
                )
        return frame
