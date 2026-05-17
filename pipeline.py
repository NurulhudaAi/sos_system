import cv2, yaml, time, logging, requests, threading, os, json, uuid
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Optional

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

    def __init__(
        self,
        webhook=None,
        cooldowns=None,
        default_cooldown=300,
        enforce_one_per_file=False,
        reset_file=None,
        reset_check_interval=1.0,
        help_dispatcher=None,
        db=None,
        snapshot_dir: Optional[str] = None,   # FIX 1: เพิ่ม snapshot_dir
    ):
        self.webhook = webhook
        self.help_dispatcher = help_dispatcher
        self.db = db
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else ALERT_DIR  # FIX 1
        self._cooldowns = cooldowns or {}
        self._default_cooldown = default_cooldown
        self._enforce_one_per_file = enforce_one_per_file
        self._file_alerted: set[str] = set()
        self._last_event: dict[str, float] = {}

        try:
            cfg = yaml.safe_load(Path("config/thresholds.yaml").read_text())
        except Exception:
            cfg = {}
        self._danger_cfg = cfg.get('danger', {})
        self._record_only_dangerous = cfg.get('record_only_dangerous', True)
        self._record_threshold = self._danger_cfg.get('record_threshold_level', 2)

        self._reset_file = Path(reset_file) if reset_file else LOG_DIR/"reset_alerts"
        self._reset_check_interval = float(reset_check_interval)
        self._reset_mtime = None
        self._reset_thread = None

        logging.basicConfig(
            filename=str(LOG_DIR/"alerts.log"),
            level=logging.WARNING,
            format="%(asctime)s | %(message)s")
        self._log = logging.getLogger("alert")

        try:
            self._reset_thread = threading.Thread(target=self._reset_watcher, daemon=True)
            self._reset_thread.start()
        except Exception:
            pass

    def _assess_alert_level(self, atype, extra):
        """Return (level_int, level_name, flags)
        level_int: 0=LOG, 1=MED, 2=HIGH, 3=CRITICAL
        """
        level = 0
        level_name = "LOG"
        flags = []
        try:
            fr = None
            if isinstance(extra, dict):
                fr = extra.get('fall_result') or extra.get('fr')
            collapse = 'balance'
            post_state = 'active_recovery'
            env_modifier = 0
            if fr:
                time_to_ground  = fr.get('time_to_ground')
                is_critical     = fr.get('is_critical')
                recovered_quickly = fr.get('recovered_quickly')
                time_lying      = fr.get('time_lying') or 0
                vel_norm        = fr.get('vel_y_norm') or fr.get('avg_vel_norm') or 0.0

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
                        collapse = 'environmental' if (vel_norm and vel_norm > 0.25) else 'balance'

                immobile_thresh = int(self._danger_cfg.get('immobile_seconds', 5))
                if time_lying >= immobile_thresh:
                    post_state = 'immobile'
                elif time_lying > 0:
                    post_state = 'limited_movement'
                else:
                    post_state = 'active_recovery'

            if isinstance(extra, dict):
                eflags = extra.get('flags') or []
                env_flags = [f for f in eflags if f in ('ROAD_ENVIRONMENT','OUTDOOR_HEAT_RISK','WORKPLACE_HAZARD')]
                if env_flags:
                    env_modifier += 1
                    flags.extend(env_flags)
                age = extra.get('age_group')
                if age in ('elderly', 'child'):
                    env_modifier += 1
                    flags.append('ELDERLY_PRIORITY' if age == 'elderly' else 'CHILD_PRIORITY')

            if collapse == 'medical':
                level = 3 if post_state in ('immobile', 'limited_movement') else 2
            elif collapse == 'environmental':
                level = 3 if post_state == 'immobile' else (2 if post_state == 'limited_movement' else 1)
            else:  # balance
                level = 2 if post_state == 'immobile' else (1 if post_state == 'limited_movement' else 0)

            level = min(3, level + env_modifier)
            level_name = ["LOG","MED","HIGH","CRITICAL"][level]
        except Exception:
            level = 0
            level_name = "LOG"
        return level, level_name, flags

    def _reset_watcher(self):
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
        self._file_alerted.clear()
        print("RUNTIME: cleared file-alerted state")

    def dispatch(self, atype: str, frame, extra=None) -> Optional[Path]:
        track  = extra.get("track_id") if isinstance(extra, dict) else None
        source = extra.get("source")   if isinstance(extra, dict) else None
        key    = f"{atype}:{track}:{source}" if (track is not None or source is not None) else atype
        now_ts = time.time()

        # ── Stumble / quick recovery → log only, skip alert ──────────────
        try:
            if isinstance(extra, dict) and (extra.get('recovered_quickly') or extra.get('skip_if_recovered')):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_path: Optional[Path] = None
                try:
                    img_path = STUMBLE_DIR / f"stumble_{atype}_{ts}.jpg"
                    cv2.imwrite(str(img_path), frame)
                except Exception:
                    img_path = None
                try:
                    with open(LOG_DIR/"stumbles.log", "a") as lf:
                        # FIX 2: ใช้ repr() แทน f-string {} เพื่อหลีก Pylance unhashable
                        lf.write(f"{datetime.now().isoformat()} | {atype} | {repr(extra)} | "
                                 f"{img_path.name if img_path else 'none'}\n")
                except Exception:
                    pass
                print(f"STUMBLE RECORDED: {img_path.name if img_path else 'none'}")
                return None
        except Exception:
            pass

        # ── One-alert-per-file enforcement ────────────────────────────────
        if self._enforce_one_per_file and source:
            try:
                if Path(str(source)).exists():
                    file_key = f"{source}:{atype}"
                    if file_key in self._file_alerted:
                        print(f"ALERT SKIPPED: {self.LABELS.get(atype, atype.upper())} (already alerted for file+event)")
                        return None
            except Exception:
                pass

        # ── Assess level ONCE ─────────────────────────────────────────────
        level, level_name, flags = self._assess_alert_level(atype, extra)  # FIX 3: คำนวณครั้งเดียว

        # merge flags into extra
        if isinstance(extra, dict):
            extra['flags'] = list(set((extra.get('flags') or []) + flags))

        if self._record_only_dangerous and level < self._record_threshold:
            print(f"ALERT SKIPPED: {atype} (level {level_name} below record threshold)")
            return None

        # ── Cooldown check ────────────────────────────────────────────────
        cooldown = self._cooldowns.get(atype, self._default_cooldown)
        if now_ts - self._last_event.get(key, 0) < cooldown:
            print(f"ALERT SKIPPED: {self.LABELS.get(atype, atype.upper())} (duplicate within cooldown)")
            return None

        # ── Save image ────────────────────────────────────────────────────
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        event_uuid = str(uuid.uuid4())
        label      = self.LABELS.get(atype, atype.upper())
        path       = self.snapshot_dir / f"{atype}_{ts}.jpg"   # FIX 1: ใช้ self.snapshot_dir

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        try:
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        except Exception:
            cv2.imwrite(str(path), frame)

        self._last_event[key] = now_ts

        # ── Write metadata JSON ───────────────────────────────────────────
        try:
            meta = {
                'ts': ts, 'event': atype, 'label': label,
                'file': path.name, 'level': int(level),
                'level_name': level_name, 'flags': flags,
                'extra': extra or {}
            }
            with open(path.with_suffix('.json'), 'w') as mf:
                json.dump(meta, mf, indent=2)
        except Exception:
            pass

        # ── MongoDB persist ───────────────────────────────────────────────
        if self.db:
            try:
                source_id  = extra.get("source_id") if isinstance(extra, dict) else None
                source_path = extra.get("source")   if isinstance(extra, dict) else None
                track_id   = extra.get("track_id")  if isinstance(extra, dict) else None
                location   = extra.get("location")  if isinstance(extra, dict) else None
                self.db.insert_sos_event(
                    event_uuid=event_uuid, event_type=atype,
                    severity=level, severity_name=level_name,
                    source_id=source_id or source_path, source_path=source_path,
                    location=location, track_id=track_id,
                    image_path=str(path),
                    meta_path=str(path.with_suffix('.json')),
                    flags=flags, extra=extra or {},
                )
            except Exception as e:
                self._log.warning(f"MongoDB event insert failed: {e}")
                print(f"⚠️  MongoDB event insert failed: {e}")

        # ── One-per-file mark ─────────────────────────────────────────────
        if self._enforce_one_per_file and source:
            try:
                if Path(str(source)).exists():
                    self._file_alerted.add(f"{source}:{atype}")
            except Exception:
                pass

        # FIX 2: ใช้ repr() แทน {} ใน f-string เพื่อหลีก Pylance unhashable warning
        self._log.warning(f"{label} | {repr(extra)} | {path.name}")
        print(f"ALERT: {label} | {path.name}")

        # ── Help request ──────────────────────────────────────────────────
        if self.help_dispatcher:
            try:
                if self.help_dispatcher.should_send_help_request(atype, level):
                    self.help_dispatcher.dispatch_help_request(
                        event_uuid=event_uuid, event_type=atype,
                        severity=level, severity_name=level_name,
                        location=(extra.get("location") if isinstance(extra, dict) else None) or "unknown",
                        source_id=extra.get("source") if isinstance(extra, dict) else None,
                        track_id=extra.get("track_id") if isinstance(extra, dict) else None,
                        image_path=str(path), frame=frame, extra=extra,
                    )
            except Exception as e:
                self._log.warning(f"Help request dispatch failed: {e}")
                print(f"⚠️  Help request error: {e}")

        return path   # FIX 3: คืน Path เสมอ (ไม่ใช่ True/False)