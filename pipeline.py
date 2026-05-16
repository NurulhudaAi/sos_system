import cv2, yaml, time, logging, requests, threading, os, json
from pathlib import Path
from collections import deque
from datetime import datetime

ALERT_DIR = Path("alerts")
LOG_DIR   = Path("logs")
ALERT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
STUMBLE_DIR = LOG_DIR/"stumbles"
STUMBLE_DIR.mkdir(exist_ok=True)

class CooldownEngine:
    def __init__(self, name, cfg):
        self.name      = name
        self.threshold = cfg.get("temporal_threshold", 0.6)
        self.cooldown  = cfg.get("cooldown_seconds",   30)
        self._buf      = deque(maxlen=cfg.get("temporal_window", 20))
        self._last     = 0.0

    def update(self, detected):
        self._buf.append(1 if detected else 0)
        if self._buf.maxlen is not None and len(self._buf) < self._buf.maxlen:
            return False
        if sum(self._buf)/len(self._buf) >= self.threshold:
            now = time.time()
            if now - self._last > self.cooldown:
                self._last = now
                return True
        return False

class ZoneManager:
    def __init__(self, path, cam_id):
        self.zones = []
        try:
            d = yaml.safe_load(Path(path).read_text())
            self.zones = d["cameras"][cam_id]["zones"]
        except Exception:
            pass

    def in_zone(self, cx, cy):
        if not self.zones:
            return True
        return any(z["x1"]<=cx<=z["x2"] and z["y1"]<=cy<=z["y2"]
                   for z in self.zones)

    def draw(self, frame):
        h,w = frame.shape[:2]
        for z in self.zones:
            cv2.rectangle(frame,
                (int(z["x1"]*w),int(z["y1"]*h)),
                (int(z["x2"]*w),int(z["y2"]*h)),(255,255,0),1)

class AlertDispatcher:
    LABELS = {
        "fall":"FALL DETECTED",
        "hand_sos":"SILENT SOS HAND",
        "fall_warning":"FALL WARNING",
    }

    def __init__(self, webhook=None, cooldowns=None, default_cooldown=300, enforce_one_per_file=False, reset_file=None, reset_check_interval=1.0, help_dispatcher=None):
        self.webhook = webhook
        self.help_dispatcher = help_dispatcher
        # per-event cooldown mapping (e.g., {'fall': 120, 'hand_sos': 60})
        self._cooldowns = cooldowns or {}
        self._default_cooldown = default_cooldown
        # enforce single alert per local file
        self._enforce_one_per_file = enforce_one_per_file
        self._file_alerted = set()
        # map event_key -> last timestamp written
        self._last_event = {}

        # load danger config
        try:
            cfg = yaml.safe_load(Path("config/thresholds.yaml").read_text())
        except Exception:
            cfg = {}
        self._danger_cfg = cfg.get('danger', {})
        self._record_only_dangerous = cfg.get('record_only_dangerous', True)
        self._record_threshold = self._danger_cfg.get('record_threshold_level', 2)

        # reset control
        self._reset_file = Path(reset_file) if reset_file else LOG_DIR/"reset_alerts"
        self._reset_check_interval = float(reset_check_interval)
        self._reset_mtime = None
        self._reset_thread = None

        logging.basicConfig(
            filename=str(LOG_DIR/"alerts.log"),
            level=logging.WARNING,
            format="%(asctime)s | %(message)s")
        self._log = logging.getLogger("alert")

        # start background watcher to allow runtime reset via presence/modification of reset file
        try:
            self._reset_thread = threading.Thread(target=self._reset_watcher, daemon=True)
            self._reset_thread.start()
        except Exception:
            pass

    def _assess_alert_level(self, atype, extra):
        """Return (level_int, level_name, flags)
        level_int: 0=LOG,1=MED,2=HIGH,3=CRITICAL
        """
        level = 0
        level_name = "LOG"
        flags = []
        try:
            fr = None
            if isinstance(extra, dict):
                fr = extra.get('fall_result') or extra.get('fr')
            # default collapse type
            collapse = 'balance'
            post_state = 'active_recovery'
            env_modifier = 0
            # basic heuristics from fall result
            if fr:
                time_to_ground = fr.get('time_to_ground')
                is_critical = fr.get('is_critical')
                recovered_quickly = fr.get('recovered_quickly')
                time_lying = fr.get('time_lying') or 0
                vel_norm = fr.get('vel_y_norm') or fr.get('avg_vel_norm') or 0.0

                # determine collapse type
                if recovered_quickly:
                    collapse = 'balance'
                    flags.append('BALANCE_STUMBLE')
                else:
                    if is_critical and time_to_ground and time_to_ground > 1.0:
                        collapse = 'medical'
                        flags.append('MEDICAL_COLLAPSE')
                    elif is_critical and time_to_ground and time_to_ground <= 1.0:
                        collapse = 'environmental'
                        flags.append('IMPACT_FALL')
                    else:
                        # fallback heuristics: high normalized velocity -> environmental
                        if vel_norm and vel_norm > 0.25:
                            collapse = 'environmental'
                        else:
                            collapse = 'balance'

                # post-fall state
                immobile_thresh = int(self._danger_cfg.get('immobile_seconds', 5))
                ext_thresh = int(self._danger_cfg.get('extended_inactivity_seconds', 30))
                if time_lying >= immobile_thresh:
                    post_state = 'immobile'
                elif time_lying > 0:
                    post_state = 'limited_movement'
                else:
                    post_state = 'active_recovery'

            # environment modifiers from extra flags (if provided)
            if isinstance(extra, dict):
                eflags = extra.get('flags') or []
                if 'ROAD_ENVIRONMENT' in eflags or 'OUTDOOR_HEAT_RISK' in eflags or 'WORKPLACE_HAZARD' in eflags:
                    env_modifier += 1
                    flags.extend([f for f in eflags if f in ('ROAD_ENVIRONMENT','OUTDOOR_HEAT_RISK','WORKPLACE_HAZARD')])
                # age modifier
                age = extra.get('age_group')
                if age in ('elderly','child'):
                    env_modifier += 1
                    if age=='elderly': flags.append('ELDERLY_PRIORITY')
                    else: flags.append('CHILD_PRIORITY')

            # base matrix mapping
            if collapse == 'medical':
                if post_state == 'immobile':
                    level = 3
                elif post_state == 'limited_movement':
                    level = 3
                else:
                    level = 2
            elif collapse == 'environmental':
                if post_state == 'immobile':
                    level = 3
                elif post_state == 'limited_movement':
                    level = 2
                else:
                    level = 1
            else: # balance
                if post_state == 'immobile':
                    level = 2
                elif post_state == 'limited_movement':
                    level = 1
                else:
                    level = 0

            # apply environment/age modifiers
            level = min(3, level + env_modifier)
            level_name = ["LOG","MED","HIGH","CRITICAL"][level]
        except Exception:
            level = 0
            level_name = "LOG"
        return level, level_name, flags

    def _reset_watcher(self):
        # watches self._reset_file; when file is created/modified, clear _file_alerted
        while True:
            try:
                if self._reset_file.exists():
                    m = os.path.getmtime(str(self._reset_file))
                    if self._reset_mtime is None or m != self._reset_mtime:
                        self._reset_mtime = m
                        self.reset_file_alerts()
                time.sleep(self._reset_check_interval)
            except Exception:
                time.sleep(self._reset_check_interval)

    def reset_file_alerts(self):
        # clear set of alerted files at runtime
        self._file_alerted.clear()
        # runtime reset performed; avoid writing to alerts.log to keep it reserved for real events
        print("RUNTIME: cleared file-alerted state")

    def dispatch(self, atype, frame, extra=None):
        # event key: prefer track_id and source when available to ensure one image per tracked entity/source
        track = None
        source = None
        if isinstance(extra, dict):
            track = extra.get("track_id")
            source = extra.get("source")
        key = f"{atype}:{track}:{source}" if track is not None or source is not None else atype
        now_ts = time.time()

        # If extra indicates this was a quick recovery (stumble), record in separate stumble log and skip normal alert
        try:
            if isinstance(extra, dict) and (extra.get('recovered_quickly') or extra.get('skip_if_recovered')):
                label = self.LABELS.get(atype, atype.upper())
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                # save image to stumbles directory for tuning (not alerts/)
                try:
                    img_name = f"stumble_{atype}_{ts}.jpg"
                    img_path = STUMBLE_DIR / img_name
                    cv2.imwrite(str(img_path), frame)
                except Exception:
                    img_path = None
                # append structured line to stumbles.log for offline analysis
                try:
                    logp = LOG_DIR / "stumbles.log"
                    with open(logp, "a") as lf:
                        lf.write(f"{datetime.now().isoformat()} | {atype} | {extra or {}} | {img_path.name if img_path else 'none'}\n")
                except Exception:
                    pass
                print(f"STUMBLE RECORDED: {img_path.name if img_path else 'none'}")
                return False
        except Exception:
            pass

        # If enforcement enabled and source is a local file that's already alerted for this event type => skip
        if self._enforce_one_per_file and source:
            try:
                if source and Path(str(source)).exists():
                    file_key = f"{str(source)}:{atype}"
                    if file_key in self._file_alerted:
                        label = self.LABELS.get(atype, atype.upper())
                        # do not write skip entries to alerts.log; only notify on console
                        print(f"ALERT SKIPPED: {label} (already alerted for file+event)")
                        return False
            except Exception:
                pass

        # assess danger level and optionally skip non-dangerous alerts
        try:
            level, level_name, new_flags = self._assess_alert_level(atype, extra)
            # merge flags into extra for logging
            if isinstance(extra, dict):
                extra_flags = extra.get('flags') or []
                extra['flags'] = list(set(extra_flags + new_flags))
            # if configured to record only dangerous events, skip below-threshold
            if self._record_only_dangerous and level < self._record_threshold:
                print(f"ALERT SKIPPED: {atype} (level {level_name} below record threshold)")
                return False
        except Exception:
            pass

        # per-event cooldown value
        cooldown = self._cooldowns.get(atype, self._default_cooldown)
        last = self._last_event.get(key, 0)
        if now_ts - last < cooldown:
            # skip saving duplicate snapshot for same event within cooldown
            label = self.LABELS.get(atype, atype.upper())
            # do not write skip entries to alerts.log; only notify on console
            print(f"ALERT SKIPPED: {label} (duplicate within cooldown)")
            return False

        ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        label = self.LABELS.get(atype, atype.upper())
        path  = ALERT_DIR/f"{atype}_{ts}.jpg"
        # write high-quality JPEG and a metadata JSON alongside the image
        try:
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        except Exception:
            cv2.imwrite(str(path), frame)
        self._last_event[key] = now_ts

        # also write metadata JSON with alert level and flags
        try:
            level, level_name, flags = self._assess_alert_level(atype, extra)
            meta = {
                'ts': ts,
                'event': atype,
                'label': label,
                'file': path.name,
                'level': int(level),
                'level_name': level_name,
                'flags': flags,
                'extra': extra or {}
            }
            meta_path = path.with_suffix('.json')
            with open(meta_path, 'w') as mf:
                json.dump(meta, mf, indent=2)
        except Exception:
            pass

        self._last_event[key] = now_ts

        # mark file+event as alerted if enforcement enabled
        if self._enforce_one_per_file and source:
            try:
                if source and Path(str(source)).exists():
                    file_key = f"{str(source)}:{atype}"
                    self._file_alerted.add(file_key)
            except Exception:
                pass

        self._log.warning(f"{label} | {extra or {{}}} | {path.name}")
        print(f"ALERT: {label} | {path.name}")

        # Send help request if dispatcher configured and triggered
        if self.help_dispatcher:
            try:
                level, level_name, flags = self._assess_alert_level(atype, extra)
                if self.help_dispatcher.should_send_help_request(atype, level):
                    import uuid
                    event_uuid = str(uuid.uuid4())
                    source_id = extra.get("source") if isinstance(extra, dict) else None
                    track_id = extra.get("track_id") if isinstance(extra, dict) else None
                    location = extra.get("location") if isinstance(extra, dict) else "unknown"
                    self.help_dispatcher.dispatch_help_request(
                        event_uuid=event_uuid,
                        event_type=atype,
                        severity=level,
                        severity_name=level_name,
                        location=location,
                        source_id=source_id,
                        track_id=track_id,
                        image_path=str(path),
                        frame=frame,
                        extra=extra
                    )
            except Exception as e:
                self._log.warning(f"Help request dispatch failed: {e}")
                print(f"⚠️  Help request error: {e}")

        return True
