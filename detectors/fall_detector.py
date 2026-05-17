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
      4. Ground    — sustained detect_ground → ground_time confirmed
      5. Confirm   — ground_time ≥ confirm_s OR geometry_fallen (3 s shortcut)
      6. Suppress  — recovered_quickly (nap/trip < recover_s) blocks is_fallen

    Config keys (all read from thresholds.yaml fall: section):
      bbox_ratio_thresh       1.3    bbox w/h > thresh → may be lying
      spine_angle_thresh      55.0   torso angle < thresh (°) → near horizontal
      warning_seconds         2.0    is_down_basic sustained → slow-fall fallback
      confirm_seconds         4.0    ground_time ≥ this → is_fallen (was 9 s)
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

    def process(self, tid: int, kps, bbox, h: int, w: int) -> dict:
        """Process one detection for track `tid` and return a result dict.

        Parameters
        ----------
        tid  : track ID (int)
        kps  : keypoints array, shape (N, 3) — [x, y, confidence]
        bbox : [x1, y1, x2, y2] in pixels
        h, w : frame height and width in pixels
        """
        x1, y1, x2, y2 = bbox
        ratio = (x2 - x1) / ((y2 - y1) + 1e-6)

        # ── Spine angle (shoulder-midpoint → hip-midpoint) ──
        ls, rs = _kp(kps, L_SHOULDER), _kp(kps, R_SHOULDER)
        lh, rh = _kp(kps, L_HIP),     _kp(kps, R_HIP)
        angle = 90.0
        has_pose = _vis(ls) and _vis(rs) and _vis(lh) and _vis(rh)
        if has_pose:
            sh = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2])
            hp = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])
            d  = sh - hp
            angle = abs(np.degrees(np.arctan2(abs(d[1]), abs(d[0]) + 1e-6)))

        now = time.time()
        cx  = (x1 + x2) / 2.0
        cy  = (y1 + y2) / 2.0
        h_pixels = float(h) if h else 1.0

        # ── Velocity history ──
        dq = self._update_hist(tid, now, cy)
        last_v, vels = self._compute_velocities(dq)
        avg_abs_v      = float(np.mean(np.abs(vels))) if vels else 0.0
        last_v_norm    = last_v    / h_pixels
        avg_abs_v_norm = avg_abs_v / h_pixels

        # ════════════════════════════════════════════════════════════════
        # STAGE 1 — Geometry signals
        # ════════════════════════════════════════════════════════════════
        is_down_basic = ratio > self.bbox_thresh or angle < self.angle_thresh

        # Strong geometry: very wide bbox AND very flat torso simultaneously.
        # Used as a keypoint-free fallback (confirm_seconds shortcut).
        strong_lying_geometry = (
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
        # STAGE 2 — Motion signals
        # ════════════════════════════════════════════════════════════════
        detect_spike = last_v_norm > self.spike_thresh_norm
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
            and since_down is not None
            and (now - since_down) >= self.warning_s
        )

        is_down = (
            (is_down_basic and (detect_spike or low_motion))
            or geometry_down
            or slow_fall_fallback   # FIX 2
        )

        # ── Spike bookkeeping ──
        if detect_spike:
            self._spike_time.setdefault(tid, now)
        else:
            # Hold spike marker for 3 s to measure time_to_ground accurately
            if tid in self._spike_time and now - self._spike_time[tid] > 3.0:
                self._spike_time.pop(tid, None)

        # ════════════════════════════════════════════════════════════════
        # STAGE 4 — Standing height calibration
        #
        # FIX 3: Original code froze stand_h from the median of the first
        # 3 frames. If those frames caught the person crouching/sitting,
        # stand_h was underestimated and detect_lying_height false-fired
        # forever.  Now we track the rolling maximum so stand_h rises
        # whenever we observe the person standing tall.
        # ════════════════════════════════════════════════════════════════
        bbox_h      = y2 - y1
        bbox_h_norm = bbox_h / h_pixels if h_pixels else 0.0

        hh = self._height_hist.get(tid)
        if hh is None:
            hh = deque(maxlen=self.history_len)
            self._height_hist[tid] = hh
        hh.append(bbox_h_norm)

        # FIX 3: rolling max — update stand_h whenever we see a taller observation
        current_max = float(max(hh))
        if current_max > 0.18:  # sanity: must look like a standing person
            prev = self._stand_height.get(tid, 0.0)
            if current_max > prev:
                self._stand_height[tid] = current_max

        stand_h = self._stand_height.get(tid)
        detect_lying_height = bool(stand_h and bbox_h_norm < stand_h * 0.6)

        # ════════════════════════════════════════════════════════════════
        # STAGE 5 — Ground contact detection
        # ════════════════════════════════════════════════════════════════
        detect_ground = (
            (low_motion and (ratio > self.bbox_thresh or (has_pose and angle < self.lying_angle_thresh)))
            or detect_lying_height
            or geometry_down
        )

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
            if tid in self._ground_time and not is_down:
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
            self._since.pop(tid, None)
            td = 0.0
            self._spike_time.pop(tid, None)
            self._ground_time.pop(tid, None)

        # ── Geometry path can set ground_time independently ──
        ground_t = self._ground_time.get(tid)
        if geometry_fallen and geom_start is not None and ground_t is None:
            self._ground_time[tid] = geom_start
            ground_t = geom_start

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
        # FIX 1: confirm_seconds default changed from 8–9 s → 4 s.
        #         The yaml value overrides this, but the in-code default
        #         is now correct.  A confirmed fall requires either:
        #           (a) ground contact sustained ≥ confirm_s seconds, OR
        #           (b) geometry_fallen (strong posture ≥ geometry_confirm_s).
        #         Quick recoveries (trip/stumble) suppress the flag.
        # ════════════════════════════════════════════════════════════════
        if ground_t:
            is_fallen = (now - ground_t) >= self.confirm_s
        else:
            is_fallen = False

        if geometry_fallen:
            is_fallen = True  # geometry shortcut (3 s of clear lying posture)

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
        is_critical = (
            (ground_t is not None and time_lying is not None and time_lying >= self.critical_s)
            or danger_lying
        )

        return {
            "is_fallen":             bool(is_fallen),
            "is_down":               bool(is_down),
            "time_down":             round(time_down_val, 1),
            "bbox_ratio":            round(ratio, 2),
            "spine_angle":           round(angle, 1),
            "vel_y":                 round(last_v, 1),
            "avg_vel":               round(avg_abs_v, 1),
            "vel_y_norm":            round(last_v_norm, 4),
            "avg_vel_norm":          round(avg_abs_v_norm, 4),
            "is_critical":           bool(is_critical),
            "spike_time":            spike_t,
            "ground_time":           ground_t,
            "time_to_ground":        time_to_ground,
            "time_lying":            time_lying,
            "danger_lying":          bool(danger_lying),
            "recovered_quickly":     bool(recovered_quickly),
            "strong_lying_geometry": bool(strong_lying_geometry),
            "geometry_time":         round(geometry_time, 1),
            "slow_fall_fallback":    bool(slow_fall_fallback),   # debug signal
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