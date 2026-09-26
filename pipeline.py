import cv2, yaml, time, logging, requests, threading, os, json, uuid
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from utils import frame_to_base64

ALERT_DIR = Path("alerts")
LOG_DIR   = Path("logs")
ALERT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
STUMBLE_DIR = LOG_DIR/"stumbles"
STUMBLE_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("pipeline")

# ─── Global ZoneManager Registry for Hot-Reloading ───────────────────────────
ZONE_MANAGERS: Dict[str, "ZoneManager"] = {}

def get_zone_manager(cam_id: str) -> Optional["ZoneManager"]:
    """Look up the active ZoneManager instance for a camera."""
    return ZONE_MANAGERS.get(str(cam_id))

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
    """Manages spatial zones (Rectangles & Polygons) with per-detector filtering and hot-reload.

    Supports:
      1. Polygons:   {"name": "stairs", "type": "polygon", "points": [[x,y],...], "detectors": ["fall"]}
      2. Rectangles: {"name": "room", "x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9, "detectors": ["hand_sos"]}
    """
    def __init__(self, path: Optional[str] = None, cam_id: str = "default", db=None):
        self.cam_id = str(cam_id)
        self.path = Path(path) if path else None
        self.db = db
        self.zones: List[Dict[str, Any]] = []
        self._compiled_polys: List[Optional[np.ndarray]] = []

        # Register instance for hot-reloads
        ZONE_MANAGERS[self.cam_id] = self
        self.load_zones()

    def load_zones(self, zones_list: Optional[List[Dict[str, Any]]] = None) -> None:
        """Load zones from a direct list, MongoDB, or fallback YAML file."""
        if zones_list is not None:
            self.reload_zones(zones_list)
            return

        loaded = None

        # 1. Try loading from MongoDB cctv_cameras if db available
        if self.db is not None:
            try:
                database = self.db._get_db() if hasattr(self.db, "_get_db") else self.db
                col = database["cctv_cameras"]
                cam = col.find_one({
                    "$or": [
                        {"code": self.cam_id},
                        {"id": self.cam_id},
                        {"name": self.cam_id},
                        {"position_note": self.cam_id}
                    ]
                })
                if cam and "zones" in cam and isinstance(cam["zones"], list):
                    loaded = cam["zones"]
            except Exception as e:
                logger.debug(f"[ZoneManager] DB load failed for {self.cam_id}: {e}")

        # 2. Fall back to YAML file
        if loaded is None and self.path and self.path.exists():
            try:
                d = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
                cams = d.get("cameras", {})
                cam_cfg = cams.get(self.cam_id) or cams.get("default")
                if cam_cfg:
                    loaded = cam_cfg.get("zones", [])
            except Exception as e:
                print(f"⚠️  [ZoneManager] Could not load zones from {self.path} (cam_id={self.cam_id!r}): {e}")

        self.reload_zones(loaded or [])

    def reload_zones(self, zones_list: List[Dict[str, Any]]) -> None:
        """Hot-swap active zones in memory safely."""
        new_zones = []
        new_compiled = []
        for z in zones_list:
            z_copy = dict(z)
            poly_pts = None
            pts = z_copy.get("points")
            if (z_copy.get("type") == "polygon" or pts is not None) and pts:
                if len(pts) >= 3:
                    poly_pts = np.array(pts, dtype=np.float32)
            new_zones.append(z_copy)
            new_compiled.append(poly_pts)

        self.zones = new_zones
        self._compiled_polys = new_compiled
        logger.info(f"[ZoneManager] Active zones for '{self.cam_id}': {len(self.zones)}")

    def in_zone(self, cx: float, cy: float, detector_type: Optional[str] = None) -> bool:
        """Check if point (cx, cy) is inside a valid zone for detector_type.

        If no zones are configured for this camera, returns True (full frame active).
        """
        if not self.zones:
            return True

        has_matching_zone = False

        for i, z in enumerate(self.zones):
            allowed = z.get("detectors")
            if allowed and isinstance(allowed, list) and len(allowed) > 0:
                if detector_type and detector_type not in allowed and "all" not in allowed:
                    continue

            has_matching_zone = True
            poly = self._compiled_polys[i] if i < len(self._compiled_polys) else None
            if poly is not None:
                # cv2.pointPolygonTest: >= 0 means inside or on edge
                if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                    return True
            else:
                x1 = float(z.get("x1", 0.0))
                y1 = float(z.get("y1", 0.0))
                x2 = float(z.get("x2", 1.0))
                y2 = float(z.get("y2", 1.0))
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    return True

        # If matching zones were checked but none contained the point
        return False

    def draw(self, frame: np.ndarray, color: tuple = (255, 255, 0)) -> None:
        """Draw zone boundaries and detector tags on the video frame."""
        if not self.zones:
            return
        h, w = frame.shape[:2]
        for i, z in enumerate(self.zones):
            name = z.get("name", f"Zone-{i+1}")
            dets = z.get("detectors")
            tag = f"{name} ({','.join(dets)})" if dets else name

            poly = self._compiled_polys[i] if i < len(self._compiled_polys) else None
            if poly is not None:
                pts_px = (poly * np.array([w, h], dtype=np.float32)).astype(np.int32)
                cv2.polylines(frame, [pts_px], isClosed=True, color=color, thickness=2)
                if len(pts_px) > 0:
                    cv2.putText(frame, tag, (pts_px[0][0], max(20, pts_px[0][1] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            else:
                x1, y1 = int(z.get("x1", 0.0) * w), int(z.get("y1", 0.0) * h)
                x2, y2 = int(z.get("x2", 1.0) * w), int(z.get("y2", 1.0) * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, tag, (x1 + 4, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

class AlertDispatcher:
    LABELS = {
        "fall":"FALL DETECTED",
        "hand_sos":"SILENT SOS HAND",
        "pose_sos":"HANDS UP SOS",
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
        self.save_local = os.getenv("SAVE_LOCAL_SNAPSHOTS", "true").lower() in ("true", "1", "yes")

        try:
            cfg = yaml.safe_load(Path("config/thresholds.yaml").read_text())
        except Exception:
            cfg = {}
        if "save_local_snapshots" in cfg:
            self.save_local = bool(cfg["save_local_snapshots"])
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
        if atype in ("hand_sos", "pose_sos"):
            return 2, "HIGH", ["SOS_GESTURE"]

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
                if self.save_local:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    img_path: Optional[Path] = None
                    try:
                        img_path = STUMBLE_DIR / f"stumble_{atype}_{ts}.jpg"
                        cv2.imwrite(str(img_path), frame)
                    except Exception:
                        img_path = None
                    try:
                        with open(LOG_DIR/"stumbles.log", "a") as lf:
                            lf.write(f"{datetime.now().isoformat()} | {atype} | {repr(extra)} | "
                                     f"{img_path.name if img_path else 'none'}\n")
                    except Exception:
                        pass
                print(f"STUMBLE RECORDED: {atype} (recovered quickly)")
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

        # ── Save image or encode Base64 ───────────────────────────────────
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        event_uuid = str(uuid.uuid4())
        label      = self.LABELS.get(atype, atype.upper())
        self._last_event[key] = now_ts

        # Convert frame to in-memory Base64 data URL
        b64_url = frame_to_base64(frame)
        path = None

        if self.save_local:
            path = self.snapshot_dir / f"{atype}_{ts}.jpg"
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            try:
                cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            except Exception:
                cv2.imwrite(str(path), frame)
            print(f"📸 [SNAPSHOT SAVED] {path}")

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

        snapshot_val = str(path) if (self.save_local and path) else b64_url

        # ── MongoDB persist ───────────────────────────────────────────────
        if self.db:
            try:
                _det_type_map = {"fall": "Fall", "hand_sos": "Gesture", "pose_sos": "Gesture", "object_event": "ObjectMissing"}
                _sev_map = {0: "Low", 1: "Medium", 2: "High", 3: "High"}
                source_id   = extra.get("source_id") if isinstance(extra, dict) else None
                source_path = extra.get("source")    if isinstance(extra, dict) else None
                track_id    = extra.get("track_id")  if isinstance(extra, dict) else None
                location    = extra.get("location")  if isinstance(extra, dict) else None
                self.db.insert_incident(
                    event_uuid     = event_uuid,
                    detection_type = _det_type_map.get(atype, "Fall"),
                    zone           = location or "Unknown-0",
                    confidence     = float(extra.get("confidence", 0.0)) if isinstance(extra, dict) else 0.0,
                    timestamp      = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    severity       = _sev_map.get(level, "High"),
                    metadata       = {
                        "cameraId": source_id or source_path,
                        "personCount": 1,
                        "trackId": track_id,
                        "imagePath": snapshot_val,
                        "originalEventType": atype,
                        "originalSeverity": level,
                        "originalSeverityName": level_name,
                        "flags": flags,
                        "extra": extra or {},
                    },
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

        log_name = path.name if path else "in-memory (base64)"
        self._log.warning(f"{label} | {repr(extra)} | {log_name}")
        print(f"ALERT: {label} | {log_name}")

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
                        image_path=snapshot_val, frame=frame, extra=extra,
                    )
            except Exception as e:
                self._log.warning(f"Help request dispatch failed: {e}")
                print(f"⚠️  Help request error: {e}")

        return snapshot_val