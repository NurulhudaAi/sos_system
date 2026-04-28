import time, cv2, numpy as np
from collections import deque

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
    """Improved fall detector with temporal confirmation, velocity spike and low-motion checks.

    Config options (keys and defaults):
      bbox_ratio_thresh: 1.6
      spine_angle_thresh: 45.0
      confirm_seconds: 8.0          # time to confirm a fall before alert
      critical_seconds: 25.0        # escalate if not recovered
      history_len: 16               # frames to keep for velocity estimation
      spike_thresh_px_s: 200.0      # downward velocity px/sec considered a fall spike
      motion_thresh_px_s: 20.0      # low motion threshold after fall
      sustain_seconds: 1.0          # seconds window to consider sustained low motion
    """
    def __init__(self, cfg):
        self.bbox_thresh        = cfg.get("bbox_ratio_thresh",  1.6)
        self.angle_thresh       = cfg.get("spine_angle_thresh", 45.0)
        self.confirm_s          = cfg.get("confirm_seconds",    8.0)
        self.critical_s         = cfg.get("critical_seconds",   25.0)
        self.history_len        = cfg.get("history_len",        16)
        # normalized thresholds (fractions of image height per second)
        self.spike_thresh_norm  = cfg.get("spike_thresh_norm",  0.18)
        self.motion_thresh_norm = cfg.get("motion_thresh_norm", 0.02)
        self.sustain_s          = cfg.get("sustain_seconds",    1.0)
        self.lying_angle_thresh = cfg.get("lying_angle_thresh", 40.0)
        self.danger_lying_s      = cfg.get("danger_lying_seconds", 10.0)
        self.recover_s           = cfg.get("recover_seconds", 5.0)

        self._since             = {}     # when "down" started per track
        self._hist              = {}     # per-track deque of (t, centroid_y)
        self._spike_time        = {}     # when a fast downward spike happened
        self._ground_time       = {}     # when ground contact (lying) detected
        # for improved ground detection using bbox height
        self._height_hist       = {}     # per-track deque of bbox heights (normed)
        self._stand_height      = {}     # estimated standing bbox height (normed)
        self._ground_candidate  = {}     # when detect_ground first observed (waiting sustain)
        # recovered_short: mark tid that lay briefly then recovered within recover_s
        self._recovered_short   = {}

    def _update_hist(self, tid, now, cy):
        dq = self._hist.get(tid)
        if dq is None:
            dq = deque(maxlen=self.history_len)
            self._hist[tid] = dq
        dq.append((now, float(cy)))
        return dq

    def _compute_velocities(self, dq): 
        if len(dq) < 2:
            return 0.0, []
        vels = []
        for i in range(1, len(dq)):
            t0, y0 = dq[i-1]
            t1, y1 = dq[i]
            dt = max(1e-6, t1 - t0)
            v = (y1 - y0) / dt
            vels.append(v)
        # last instantaneous velocity
        last_v = vels[-1]
        return last_v, vels

    def process(self, tid, kps, bbox, h, w):
        x1,y1,x2,y2 = bbox
        ratio = (x2-x1) / ((y2-y1) + 1e-6)

        ls,rs = _kp(kps,L_SHOULDER), _kp(kps,R_SHOULDER)
        lh,rh = _kp(kps,L_HIP),     _kp(kps,R_HIP)
        angle = 90.0
        has_pose = _vis(ls) and _vis(rs) and _vis(lh) and _vis(rh)
        if has_pose:
            sh = np.array([(ls[0]+rs[0])/2,(ls[1]+rs[1])/2])
            hp = np.array([(lh[0]+rh[0])/2,(lh[1]+rh[1])/2])
            d  = sh - hp
            angle = abs(np.degrees(
                np.arctan2(abs(d[1]), abs(d[0])+1e-6)))

        now = time.time()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # update history and compute velocities (y grows downward)
        dq = self._update_hist(tid, now, cy)
        last_v, vels = self._compute_velocities(dq)  # pixels/sec
        avg_abs_v = float(np.mean(np.abs(vels))) if vels else 0.0

        # normalize velocities by image height to be resolution-independent
        h_pixels = float(h) if h else 1.0
        last_v_norm = last_v / h_pixels
        avg_abs_v_norm = avg_abs_v / h_pixels

        # basic geometric check
        is_down_basic = ratio > self.bbox_thresh or angle < self.angle_thresh

        # detect downward spike (fast downward movement) using normalized thresholds
        detect_spike = last_v_norm > self.spike_thresh_norm

        # check sustained low motion in recent sustain_s seconds (normalized)
        total_span = 0.0
        low_motion = False
        if len(dq) >= 2:
            # sum durations from the end until covering sustain_s
            span = 0.0
            idx = len(dq) - 1
            abs_vels = []
            while idx > 0 and span < self.sustain_s:
                t0 = dq[idx-1][0]
                t1 = dq[idx][0]
                span += (t1 - t0)
                v_px_s = abs((dq[idx][1]-dq[idx-1][1]) / max(1e-6, t1 - t0))
                abs_vels.append(v_px_s / h_pixels)
                idx -= 1
            if abs_vels:
                mean_recent_norm = float(np.mean(abs_vels))
                low_motion = mean_recent_norm < self.motion_thresh_norm

        # final down decision requires geometry + (spike OR low motion)
        is_down = is_down_basic and (detect_spike or low_motion)

        # track spike time
        if detect_spike:
            if tid not in self._spike_time:
                self._spike_time[tid] = now
        else:
            # keep spike up to 3s to allow measuring time_to_ground
            if tid in self._spike_time and now - self._spike_time[tid] > 3.0:
                self._spike_time.pop(tid, None)

        # enhanced ground detection: include bbox-height-based lying detection
        bbox_h = (y2 - y1)
        bbox_h_norm = bbox_h / h_pixels if h_pixels else 0.0
        # update height history
        hh = self._height_hist.get(tid)
        if hh is None:
            hh = deque(maxlen=self.history_len)
            self._height_hist[tid] = hh
        hh.append(bbox_h_norm)
        # estimate standing height from median of initial frames
        if tid not in self._stand_height and len(hh) >= 3:
            median_h = float(np.median(list(hh)))
            # only consider as standing if reasonably tall
            if median_h > 0.18:
                self._stand_height[tid] = median_h
        stand_h = self._stand_height.get(tid)

        detect_lying_height = False
        if stand_h:
            detect_lying_height = bbox_h_norm < (stand_h * 0.6)

        # detect ground contact (lying): low motion + geometry OR lying by height
        detect_ground = (low_motion and (ratio > self.bbox_thresh or (has_pose and angle < self.lying_angle_thresh))) or detect_lying_height

        # require sustain_seconds of detect_ground before confirming ground_time
        if detect_ground:
            if tid not in self._ground_candidate:
                self._ground_candidate[tid] = now
            else:
                if tid not in self._ground_time and (now - self._ground_candidate[tid]) >= self.sustain_s:
                    self._ground_time[tid] = self._ground_candidate[tid]
        else:
            # clear candidate and ground_time if recovered
            self._ground_candidate.pop(tid, None)
            if tid in self._ground_time and not is_down:
                # check quick recovery: if lying duration < recover_s -> mark recovered_short
                ground_start = self._ground_time.pop(tid, None)
                try:
                    if ground_start is not None:
                        lying_d = now - ground_start
                        if lying_d < self.recover_s:
                            self._recovered_short[tid] = now
                except Exception:
                    pass

        # update since/down timer
        if is_down:
            self._since.setdefault(tid, now)
            td = now - self._since[tid]
        else:
            self._since.pop(tid, None)
            td = 0.0
            # clear transient state if recovered
            if not is_down and tid in self._spike_time:
                self._spike_time.pop(tid, None)
            if not is_down and tid in self._ground_time:
                self._ground_time.pop(tid, None)

        # compute timings
        spike_t = self._spike_time.get(tid)
        ground_t = self._ground_time.get(tid)
        time_to_ground = None
        time_lying = None
        danger_lying = False
        if spike_t and ground_t:
            time_to_ground = round(max(0.0, ground_t - spike_t), 2)
        if ground_t:
            time_lying = round(max(0.0, now - ground_t), 1)
            if time_lying >= self.danger_lying_s:
                danger_lying = True

        # Conservative fall confirmation: require ground contact and sustained lying
        # is_fallen only when ground detected and lying duration >= confirm_s
        if ground_t:
            is_fallen = (now - ground_t) >= self.confirm_s
        else:
            is_fallen = False

        # If we recently recovered quickly (laying < recover_s), suppress fall
        recovered_quickly = False
        if tid in self._recovered_short:
            # expire mark after some time (recover_s*2)
            tmark = self._recovered_short.get(tid)
            if tmark and (now - tmark) < (self.recover_s * 2):
                recovered_quickly = True
            else:
                # expired
                self._recovered_short.pop(tid, None)

        if recovered_quickly:
            is_fallen = False

        # compute time_down: prefer time since ground contact if available
        if ground_t:
            time_down_val = now - ground_t
        else:
            time_down_val = td

        # escalate if critical by config or danger lying time
        is_critical = (ground_t is not None and time_lying is not None and time_lying >= self.critical_s) or danger_lying

        return {
            "is_fallen":   bool(is_fallen),
            "is_down":     bool(is_down),
            "time_down":   round(time_down_val, 1),
            "bbox_ratio":  round(ratio, 2),
            "spine_angle": round(angle, 1),
            "vel_y":       round(last_v, 1),
            "avg_vel":     round(avg_abs_v, 1),
            "vel_y_norm":  round(last_v_norm, 4),
            "avg_vel_norm":round(avg_abs_v_norm, 4),
            "is_critical": bool(is_critical),
            "spike_time":  spike_t,
            "ground_time": ground_t,
            "time_to_ground": time_to_ground,
            "time_lying":  time_lying,
            "danger_lying": bool(danger_lying),
            "recovered_quickly": bool(recovered_quickly),
        }

    def draw(self, frame, tid, res, bbox, kps, conf=0.0):
        x1,y1,x2,y2 = [int(v) for v in bbox]
        if res.get("is_critical"):
            c = (0,0,220)
        elif res.get("is_fallen"):
            c = (0,0,200)
        elif res.get("is_down"):
            c = (0,140,255)
        else:
            c = (0,200,0)

        cv2.rectangle(frame,(x1,y1),(x2,y2),c,2)
        cv2.putText(frame,f"ID:{tid} {conf:.0%}",
                    (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.55,c,1)
        if res["is_down"] and res["time_down"] > 0:
            cv2.putText(frame,f"DOWN {res['time_down']}s",
                        (x1,y2+18),cv2.FONT_HERSHEY_SIMPLEX,0.55,c,2)
        # show velocity for debugging
        if "vel_y" in res:
            cv2.putText(frame,f"vel:{res['vel_y']}px/s avg:{res.get('avg_vel',0)}",
                        (x1,y2+36),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
        if res.get("is_critical"):
            cv2.putText(frame,"CRITICAL",(x1,y2+56),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,200),2)

        for a,b in [
            (L_SHOULDER,R_SHOULDER),
            (L_SHOULDER,L_ELBOW),(L_ELBOW,L_WRIST),
            (R_SHOULDER,R_ELBOW),(R_ELBOW,R_WRIST),
            (L_SHOULDER,L_HIP),(R_SHOULDER,R_HIP),(L_HIP,R_HIP),
            (L_HIP,L_KNEE),(L_KNEE,L_ANKLE),
            (R_HIP,R_KNEE),(R_KNEE,R_ANKLE),
        ]:
            ka,kb = _kp(kps,a),_kp(kps,b)
            if _vis(ka) and _vis(kb):
                cv2.line(frame,
                    (int(ka[0]),int(ka[1])),
                    (int(kb[0]),int(kb[1])),(160,160,160),1)
        return frame
