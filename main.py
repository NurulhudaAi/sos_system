#!/usr/bin/env python3
"""
main_production.py — SOS + Object Guardian — Production v2 (Patched)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fixes applied (Audit รอบ 3):
  [B1] ลบ alert_logger.log_sos_event() ออกจาก main — ป้องกัน double-write
  [B2] HandSOSDetector ใช้ per-track hand_states[tid] แทน global _state
  [B3] sources.yaml id ซ้ำกัน 3 ตัว — แก้เป็น unique id
  [W3] import uuid ย้ายขึ้น top-level
  [W5] Object Guardian alerts ส่ง insert_object_event() ใน pipeline
  [W6] ลบ _get_db import ที่ไม่ได้ใช้

Fixes applied (Audit รอบ 4 — "fall not detected sometimes on RTSP"):
  [F1] SimpleTracker: IOU-only matching (thresh=0.3) lost the track ID
       mid-fall because bbox shape changes fast (tall→wide) between
       consecutive frames, dropping IOU below threshold right when a
       fall happens. Added a 2nd-pass centroid-distance fallback match
       so the same physical person keeps their track_id through a fall,
       instead of silently resetting all of fall_d's accumulated state
       (_since/_ground_time/etc.) onto a brand-new tid.
  [F2] Pass posture_class/posture_conf from /detect_all (if the optional
       custom laying/standing model is configured server-side) through
       to fall_d.process(), so FallDetector can use it as a direct
       geometry-independent lying signal.
"""
import uuid  # [W3] top-level
import cv2, sys, time, yaml, torch, requests, os
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timezone
from math import ceil
import numpy as np, multiprocessing, logging
from typing import Optional, Dict, Any  # [QA] was missing → NameError on Python < 3.14

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
logger = logging.getLogger("main")

import ctypes, glob
def silence_stream_noise():
    """Silences noisy C-level FFmpeg decoders and MediaPipe warnings in the terminal."""
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
    os.environ["OPENCV_LOG_LEVEL"] = "OFF"
    os.environ["GLOG_minloglevel"] = "2"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    try:
        cv_dir = os.path.dirname(cv2.__file__)
        dylibs = glob.glob(os.path.join(cv_dir, ".dylibs", "libavutil*.dylib"))
        if dylibs:
            avutil = ctypes.CDLL(dylibs[0])
            avutil.av_log_set_level(-8)  # AV_LOG_QUIET
    except Exception:
        pass

silence_stream_noise()

# ──── Initialize environment (MUST BE FIRST) ────────────────────────────────
from config.env_manager import init_env
if not init_env(require_edit=False):
    print("Environment initialization failed. Exiting.")
    sys.exit(1)
# ────────────────────────────────────────────────────────────────────────────

import re

def _safe_name(s: str) -> str:
    """Sanitize a directory name while preserving Unicode (keeps Thai)."""
    if not s:
        return "unknown"
    name = re.sub(r"[\\/\x00-\x1f]+", "_", str(s)).strip()
    name = name.replace(" ", "_")
    name = re.sub(r"_+", "_", name)
    return name

from detectors.fall_detector            import FallDetector
from detectors.hand_sos_detector        import HandSOSDetector
from detectors.hands_over_head_detector import HandsOverHeadDetector
from detectors.object_guardian          import ObjectGuardian
from pipeline                    import CooldownEngine, ZoneManager, AlertDispatcher
from help_request_dispatcher     import HelpRequestDispatcher
from utils                       import preprocess, Visualizer, add_sos_badge, create_bbox_snapshot_base64, save_annotated_snapshot
from vlc_stream                  import VLCStreamManager
from stream_reader               import LatestFrameReader
import database as db_module
from database                    import insert_incident, insert_object_event  # [W6] ลบ _get_db ที่ไม่ใช้

cfg         = yaml.safe_load((ROOT / "config/thresholds.yaml").read_text())
GEN         = cfg.get("general", {})
STREAM_CFG  = cfg.get("streaming", {})
SAMPLING_FPS= STREAM_CFG.get("fps", 10)
ROLL_WIN    = STREAM_CFG.get("rolling_window", 90)
CONF_WIN    = STREAM_CFG.get("confirmation_window", 30)
TRIG_DELTA  = STREAM_CFG.get("trigger_angle_delta", 30)
TRIG_FRAMES = STREAM_CFG.get("trigger_frames", 5)
LYING_ANG   = STREAM_CFG.get("lying_angle_thresh", 45)
W, H        = 1920, 1080
SKIP        = GEN.get("frame_skip", 2)
DET_URL     = os.getenv("DETECTOR_URL", GEN.get("detector_url", "http://127.0.0.1:8000"))
DEVICE      = ("mps" if torch.backends.mps.is_available() else
               "cuda" if torch.cuda.is_available() else "cpu")
print(f"[main] Device: {DEVICE}")

# [F1] Tracker tuning — see SimpleTracker docstring below.
TRACKER_IOU_THRESH        = float(GEN.get("tracker_iou_thresh", 0.3))
TRACKER_CENTER_DIST_NORM  = float(GEN.get("tracker_center_dist_norm", 0.12))  # fraction of frame diagonal
TRACKER_MAX_LOST          = int(GEN.get("tracker_max_lost", max(15, int(SAMPLING_FPS * 1.5))))


class _W:
    def __init__(self,v): self.val=v
    def cpu(self): return self
    def numpy(self): return np.array(self.val)
    def __float__(self):
        try: return float(self.val)
        except Exception: return float(np.array(self.val))
    def __int__(self): return int(self.__float__())
    def __repr__(self): return f"_W({self.val!r})"

class BoxesWrapper:
    def __init__(self,xyxy,confs,ids):
        self.xyxy=[_W(x) for x in xyxy]; self.conf=[_W(c) for c in confs]
        self.id=[_W(i) for i in ids] if ids else None

class KeypointsWrapper:
    def __init__(self,data): self.data=[_W(d) for d in data] if data else None

from tracker import HungarianTracker, SimpleTracker, SpatialAlertDeduplicator


def _api(endpoint, frame, timeout=5):
    _,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,80])
    try:
        r=requests.post(DET_URL+endpoint,files={"image":("f.jpg",buf.tobytes(),"image/jpeg")},timeout=timeout)
        if r.status_code==200: return r.json()
    except Exception: pass
    return {}

def _are_both_arms_over_head(kps, conf_thresh=0.25):
    """Checks if both arms are genuinely raised overhead in a wide V-shape (Pose SOS).
    Requires wrists to be spread apart, preventing Hand SOS or holding objects from being suppressed.
    """
    if len(kps) < 11:
        return False
    l_wr, r_wr = kps[9], kps[10]
    nose = kps[0]
    l_sh, r_sh = kps[5], kps[6]
    if float(l_wr[2]) < conf_thresh or float(r_wr[2]) < conf_thresh:
        return False
    if float(l_sh[2]) < conf_thresh or float(r_sh[2]) < conf_thresh:
        return False

    shoulder_y = min(float(l_sh[1]), float(r_sh[1]))
    shoulder_w = abs(float(l_sh[0]) - float(r_sh[0]))
    wrist_sep = abs(float(l_wr[0]) - float(r_wr[0]))

    # Must have genuine V-shape separation (arms spread apart)
    min_sep = max(18.0, 0.55 * shoulder_w)
    if wrist_sep < min_sep:
        return False

    head_ref = float(nose[1]) if float(nose[2]) >= conf_thresh else (shoulder_y - 15.0)

    # Both wrists must be near/above head AND clearly above shoulders
    return (
        float(l_wr[1]) <= head_ref + 5.0
        and float(r_wr[1]) <= head_ref + 5.0
        and float(l_wr[1]) <= shoulder_y - 12.0
        and float(r_wr[1]) <= shoulder_y - 12.0
    )



def main(src:str, port:int=8081, location:str="", cam_id:str="", use_vlc:bool=False, loop:bool=False, signal_config:Optional[Dict[str, Any]]=None):
    if isinstance(src, str):
        src = src.strip().strip("'\"").lstrip("=").strip().strip("'\"")
    hand_cfg = cfg.get("hand_sos", {})
    source_p = Path(src)
    is_file = source_p.is_file()
    source_id = str(source_p.resolve()) if is_file else str(src)

    vlc_mgr=None
    if is_file and not use_vlc:
        url = str(source_p.resolve())
        print(f"[main] Reading video file directly: {url}")
    elif src.startswith("http") or src.startswith("rtsp"):
        url=src
    else:
        vlc_mgr=VLCStreamManager(src=src,width=W,height=H,fps=SAMPLING_FPS,port=port)
        try:
            url=vlc_mgr.start()
        except RuntimeError as e:
            print(f"\n❌ [VLC] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return

        print(f"[VLC] Waiting for stream at {url} ...")
        for attempt in range(10):
            if vlc_mgr.health_check():
                print(f"[VLC] Stream confirmed ready (attempt {attempt+1})")
                break
            time.sleep(1)
        else:
            print(f"⚠️  [VLC] Stream not responding after 10s — continuing anyway")

    # [QA-P0-1] Live sources (RTSP / HTTP / VLC) use a background reader
    # that keeps only the newest frame → no buffer build-up / lag / smear.
    # Video files are read sequentially (every frame matters for eval).
    is_live = not (is_file and not vlc_mgr)
    cap = None; reader = None; file_fps = 25.0; file_frame_idx = 0
    if is_live:
        reader = LatestFrameReader(url, name=source_id[-30:])
        if not reader.start():
            print(f"[cap] Cannot open: {url}")
            if vlc_mgr: vlc_mgr.stop()
            return
    else:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"[cap] Cannot open: {url}")
            return
        file_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if file_fps <= 1 or file_fps > 240: file_fps = 25.0

    print(f"[{source_id[-30:]}] Connected | location={location or '?'} | cam_id={cam_id or 'default'}")

    tracker = SimpleTracker(frame_w=W, frame_h=H, max_lost=TRACKER_MAX_LOST)  # [F1]
    fall_d  =FallDetector(cfg.get("fall",{}))
    hand_d  =HandSOSDetector(cfg.get("hand_sos",{}))
    head_d  =HandsOverHeadDetector(cfg.get("hands_over_head",{}))
    obj_grd =ObjectGuardian({**cfg.get("object_guardian",{}), "alert_dir":"alerts"})
    active_cam_id = cam_id or location or "default"
    zones   =ZoneManager(str(ROOT / "config/zones.yaml"), cam_id=active_cam_id, db=db_module)
    viz     =Visualizer()
    spatial_dedup = SpatialAlertDeduplicator(radius_norm=0.12, default_cooldown_sec=30.0)

    # ── Signal Config: Incidents Enabled/Disabled from DB (cctv_cameras) ─────
    if signal_config is None and db_module:
        current_signal_cfg = db_module.get_camera_signal_config(active_cam_id)
    elif isinstance(signal_config, dict):
        current_signal_cfg = dict(signal_config)
    else:
        current_signal_cfg = {}

    fall_active = bool(current_signal_cfg.get("fall", True))
    hand_active = bool(current_signal_cfg.get("hand_sos", True))
    pose_active = bool(current_signal_cfg.get("pose_sos", current_signal_cfg.get("hand_sos", True)))
    obj_active  = bool(current_signal_cfg.get("object", True))
    last_cfg_refresh_ts = time.time()

    print(f"[{source_id[-30:]}] [CONFIG] Incident Config from DB: "
          f"Fall={'[ON]' if fall_active else '[OFF]'} | "
          f"Hand SOS={'[ON]' if hand_active else '[OFF]'} | "
          f"Pose SOS={'[ON]' if pose_active else '[OFF]'} | "
          f"Object={'[ON]' if obj_active else '[OFF]'}")

    alert_cd     = GEN.get("alert_cooldown_seconds", cfg.get("fall",{}).get("cooldown_seconds",120))
    snapshot_dir = str(ROOT / "logs" / "snapshots")
    save_local_snapshots = bool(cfg.get("save_local_snapshots", True))
    if os.getenv("SAVE_LOCAL_SNAPSHOTS"):
        save_local_snapshots = os.getenv("SAVE_LOCAL_SNAPSHOTS").lower() in ("true", "1", "yes")

    webhook_url = os.getenv("HELP_WEBHOOK_URL")
    if not webhook_url:
        print("⚠️  HELP_WEBHOOK_URL ไม่ได้ตั้งค่า — help request จะไม่ทำงาน")

    # [FIX] db= was missing → help_requests collection stayed empty forever
    help_disp = HelpRequestDispatcher(webhook_url=webhook_url or "", db=db_module)

    disp=AlertDispatcher(
        cooldowns={"fall":cfg.get("fall",{}).get("cooldown_seconds",alert_cd),
                   "hand_sos":cfg.get("hand_sos",{}).get("cooldown_seconds",alert_cd),
                   "pose_sos":cfg.get("hands_over_head",{}).get("cooldown_seconds",alert_cd)}, 
        default_cooldown=alert_cd,
        enforce_one_per_file=GEN.get("one_alert_per_file",False),
        snapshot_dir=snapshot_dir,
        help_dispatcher=help_disp)

    # [B2] per-track state — ไม่ใช้ hand_d._state global อีกต่อไป
    hand_states: dict[int, int] = {}
    hand_miss:   dict[int, int] = {}
    hand_first_seen: dict[int, float] = {}
    hand_temporal_window = max(1, int(hand_cfg.get("temporal_window", 10)))
    hand_temporal_threshold = min(1.0, max(0.0, float(hand_cfg.get("temporal_threshold", 0.20))))
    hand_temporal_hits_required = max(1, ceil(hand_temporal_window * hand_temporal_threshold))
    hand_min_bbox_area_norm = max(0.0, float(hand_cfg.get("min_hand_bbox_area_norm", 0.0)))
    hand_min_track_age_seconds = max(0.0, float(hand_cfg.get("min_track_age_seconds", 0.8)))
    hand_cooldown_seconds = max(0.0, float(hand_cfg.get("cooldown_seconds", alert_cd)))
    hand_recent_sos = defaultdict(lambda: deque(maxlen=hand_temporal_window))
    # [FIX-FP] นับ state=3 ต่อเนื่องต่อ track
    hand_consec3: dict[int, int] = {}
    MIN_CONSEC3 = max(1, int(hand_cfg.get("min_consec_sos_frames", 1)))
    hand_state1_time: dict[int, float] = {}
    GRACE_FRAMES = 8

    hand_ev={}; hand_bc={}; hand_bf={}; hand_bt={}
    _hand_last_event_ts: dict[int, float] = {}  # per-tid: cooldown timestamp
    _pose_last_event_ts: dict[int, float] = {}  # per-tid: cooldown timestamp
    hand_step_time: dict[int, float] = {}       # per-tid: step transition timestamp
    pose_cooldown_seconds = max(0.0, float(cfg.get("hands_over_head", {}).get("cooldown_seconds", alert_cd)))
    fall_ev={}; fall_bc={}; fall_bf={}; fall_bt={}
    _fall_last_event_ts = {}  # per-tid: timestamp of last confirmed fall alert
    FALL_HYSTERESIS_FACTOR = cfg.get("fall", {}).get("fall_hysteresis_factor", 2.0)
    FALL_CONFIRM_S = cfg.get("fall", {}).get("confirm_seconds", 4.0)
    # [QA-P0-5] dedup window was only confirm_s*2 (=8s) → re-alerted every ~8s
    # while a person lay still with a flickering is_fallen. Honor cooldown.
    FALL_COOLDOWN_S = float(cfg.get("fall", {}).get("cooldown_seconds", 120))
    FALL_DEBUG_CSV = os.getenv("FALL_DEBUG_CSV", "").lower() in ("1", "true", "yes")
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    last_stats_ts = time.time(); frames_done = 0
    s_states=defaultdict(lambda:{"angle_hist":deque(maxlen=TRIG_FRAMES),
                                  "motion_hist":deque(maxlen=TRIG_FRAMES),
                                  "triggered":False,"trigger_time":None,"frames_in_trigger":0})
    track_prev_center: dict[int, tuple[float, float, float]] = {}
    retry_count = 0
    max_retries = 10
    n=0
    try:
        while True:
            if is_live:
                ret, raw, frame_ts = reader.read(timeout=5.0)
                if not ret:
                    retry_count += 1
                    print(f" [{source_id[-30:]}] No frame for 5s (stall {retry_count}/{max_retries}) — reader is reconnecting")
                    if vlc_mgr and not vlc_mgr.is_alive():
                        print(" [VLC] Process died — restarting ...")
                        try:
                            url = vlc_mgr.restart()
                            reader.reopen(url)
                        except Exception as restart_err:
                            print(f"❌ [VLC] Restart failed: {restart_err}")
                    if retry_count > max_retries:
                        print(f" [{source_id[-30:]}] Max retries ({max_retries}) reached — exiting")
                        break
                    continue
                retry_count = 0
                n += 1  # reader already drops stale frames → no SKIP for live
            else:
                ret, raw = cap.read()
                if not ret:
                    if loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print(f"[{source_id[-30:]}] Video file reached end of stream — completed.")
                    break
                file_frame_idx += 1
                n += 1
                if n % max(1, SKIP) != 0: continue
                # [QA-P0-4] video time, monotonic across loops
                frame_ts = file_frame_idx / file_fps

            frame=preprocess(raw,W,H)
            h,w=frame.shape[:2]
            clean_frame=frame.copy()
            # [QA-P0-3] zones are drawn AFTER detection (was drawn onto the
            # frame sent to YOLO/MediaPipe, corrupting keypoints/hands)

            frames_done += 1
            if is_live and time.time() - last_stats_ts >= 30.0:
                el = time.time() - last_stats_ts
                print(f"[{source_id[-30:]}] proc {frames_done/el:.1f} fps | "
                      f"dropped {reader.dropped} | reconnects {reader.reconnects} | "
                      f"latency {time.time()-frame_ts:.2f}s")
                last_stats_ts = time.time(); frames_done = 0

            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # [RTSP FIX] CLAHE preprocessing — applied per-person crop below
            # (full-frame hand detection removed: MediaPipe can't find small
            # hands in a 1920×1080 RTSP frame; per-person crop + resize ≥256px
            # is required — same approach as eval_harness_full.py)
            lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
            lab[:,:,0]=_clahe.apply(lab[:,:,0])
            frame_clahe=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)

            # Periodic check for dynamic signal_config updates in DB (every 10s)
            now_loop_t = time.time()
            if db_module and (now_loop_t - last_cfg_refresh_ts > 10.0):
                last_cfg_refresh_ts = now_loop_t
                try:
                    new_sig_cfg = db_module.get_camera_signal_config(active_cam_id)
                    if new_sig_cfg != current_signal_cfg:
                        current_signal_cfg = new_sig_cfg
                        fall_active = bool(current_signal_cfg.get("fall", True))
                        hand_active = bool(current_signal_cfg.get("hand_sos", True))
                        pose_active = bool(current_signal_cfg.get("pose_sos", current_signal_cfg.get("hand_sos", True)))
                        obj_active  = bool(current_signal_cfg.get("object", True))
                        print(f"[{source_id[-30:]}] [REFRESH] Live Refreshed Incident Config: "
                              f"Fall={'[ON]' if fall_active else '[OFF]'} | "
                              f"Hand SOS={'[ON]' if hand_active else '[OFF]'} | "
                              f"Pose SOS={'[ON]' if pose_active else '[OFF]'} | "
                              f"Object={'[ON]' if obj_active else '[OFF]'}")
                except Exception:
                    pass

            now_time=time.strftime("%Y-%m-%d %H:%M:%S")
            resp=_api("/detect_all",frame)
            pdets=resp.get("people",[])
            odets=resp.get("objects",[])

            # ── [FALL-DEBUG] no-person vs not-confirmed diagnostic ──
            if not pdets:
                logger.debug(
                    "[FALL-DEBUG] frame %d (%.1fs): "
                    "no person bbox detected at all (detector returned 0 people)",
                    n, n / max(1, SAMPLING_FPS),
                )

            # ── People tracking ──────────────────────────────────────────
            if pdets:
                assigned=tracker.update(pdets)
                boxes=BoxesWrapper([d["bbox"] for d in assigned],
                                   [d.get("conf",0.0) for d in assigned],
                                   [d.get("track_id") for d in assigned])
                kpts=KeypointsWrapper([d.get("keypoints",[]) for d in assigned])
                # [F2] posture_class/posture_conf from the optional custom
                # laying/standing model (present only if configured server-
                # side in model_server.py; empty dicts otherwise → no-op).
                posture_by_idx = [
                    (d.get("posture_class"), float(d.get("posture_conf", 0.0)))
                    for d in assigned
                ]
            else:
                assigned=[]
                boxes=None; kpts=None; posture_by_idx=[]

            # ── Object Guardian (Gated by DB signal_config) ──────────────
            if obj_active:
                for oa in obj_grd.update(clean_frame, odets, assigned, source_id=source_id, location=location, t_sec=frame_ts):
                    obbox = oa.get("bbox") or [0, 0, 0, 0]
                    ocx, ocy = (obbox[0] + obbox[2]) / 2 / max(1, w), (obbox[1] + obbox[3]) / 2 / max(1, h)
                    if not zones.in_zone(ocx, ocy, detector_type="object_guardian"):
                        continue
                    # [W5] ส่ง insert_object_event ให้ตรงกับ signature จริงใน database.py
                    try:
                        insert_object_event(
                            event_type         = oa.get("event_type", "object_event"),
                            source_id          = source_id,
                            location           = location,
                            track_id           = oa.get("track_id"),
                            person_track_id    = oa.get("person_track_id") or oa.get("owner_track_id") or oa.get("suspect_track_id"),
                            class_name         = oa.get("class_name", ""),
                            confidence         = float(oa.get("confidence", 0.0)),
                            bbox               = oa.get("bbox"),
                            image_path         = oa.get("image_path"),
                            seconds_unattended = float(oa.get("seconds_unattended", 0.0)),
                            alert_raised       = bool(oa.get("alert_raised", True)),
                            meta               = oa,
                        )
                    except Exception as e:
                        print(f"[DB] object_event insert error: {e}")
                obj_grd.draw(frame)

            # [W2] cleanup tracks ที่หายไปจากเฟรม
            # [FIX-tracking] เดิมเช็คจาก `active` (เฉพาะ track_id ที่เห็นในเฟรมปัจจุบันเฟรม
            # เดียว) ทำให้ hand_states/fall state ถูกล้างทิ้งทันทีที่ detect พลาดแม้แค่ 1
            # เฟรม (เช่น motion blur ตอนคนลุกขึ้นเดินเร็วๆ) — ทั้งที่ SimpleTracker เองยัง
            # ไม่ทันลบ track (มี grace period เก็บ track ไว้จนกว่าจะ lost ติดกัน >5 เฟรม)
            # ผลคือ progress ของ state machine (เช่น hand_sos state 2 ใกล้จะถึง 3) หายไป
            # กลางคันบ่อยๆ โดยไม่จำเป็น → เปลี่ยนมาเช็คจาก tracker.tracks แทน เพื่อให้
            # cleanup ยึด grace period เดียวกับที่ tracker ใช้จริง
            if boxes and boxes.id:
                alive = set(tracker.tracks.keys())
                dropped = (set(hand_states.keys()) | set(head_d._since.keys())) - alive
                for tid in dropped:
                    fall_d.cleanup_track(tid)  # [FIX] previously missing — stale fall
                    head_d.cleanup_track(tid)
                    for d in [hand_states, hand_miss, hand_ev, fall_ev, hand_bc, hand_bf,
                          hand_bt, fall_bc, fall_bf, fall_bt, hand_first_seen, hand_consec3,
                          _hand_last_event_ts, _pose_last_event_ts, hand_step_time]:
                        d.pop(tid, None)
                    hand_recent_sos.pop(tid, None)

            if boxes and kpts and kpts.data:
                all_frame_tracks = [
                    {
                        "track_id": int(boxes.id[j].cpu()) if boxes.id else j,
                        "bbox": boxes.xyxy[j].cpu().numpy().tolist(),
                    }
                    for j in range(len(boxes.xyxy))
                ]
                for i in range(len(boxes.xyxy)):
                    bbox=boxes.xyxy[i].cpu().numpy().tolist()
                    conf=float(boxes.conf[i].cpu())
                    tid=int(boxes.id[i].cpu()) if boxes.id else i
                    if i>=len(kpts.data): continue
                    kp=kpts.data[i].cpu().numpy()
                    x1,y1,x2,y2=bbox
                    cx, cy = (x1+x2)/2/max(1, w), (y1+y2)/2/max(1, h)
                    if not zones.in_zone(cx, cy): continue

                    # ── Fall Detection (Zone-gated & DB Signal-Config gated) ──
                    posture_class, posture_conf = posture_by_idx[i] if i < len(posture_by_idx) else (None, 0.0)
                    in_fall_zone = zones.in_zone(cx, cy, detector_type="fall")
                    if fall_active and in_fall_zone:
                        fr=fall_d.process(tid,kp,bbox,h,w,
                                           posture_class=posture_class,
                                           posture_conf=posture_conf,  # [F2]
                                           timestamp=frame_ts)          # [QA-P0-4]
                    else:
                        fr={"is_fallen":False, "danger_lying":False, "is_critical":False}

                    # ── Streaming trigger ────────────────────────────────
                    try:
                        st=s_states[tid]
                        angle=float(fr.get("spine_angle") or 0.0)
                        avn=float(fr.get("avg_vel_norm") or 0.0)
                        st["angle_hist"].append(angle); st["motion_hist"].append(abs(avn))
                        trig=False
                        if len(st["angle_hist"])>=TRIG_FRAMES:
                            if (max(st["angle_hist"])-min(st["angle_hist"]))>=TRIG_DELTA: trig=True
                        mth=cfg.get("fall",{}).get("motion_thresh_norm",0.02)
                        if len(st["motion_hist"])>=2:
                            if st["motion_hist"][-2]>mth*3 and st["motion_hist"][-1]<mth and angle>=LYING_ANG: trig=True
                        if sum(1 for a in st["angle_hist"] if a>=LYING_ANG)>=TRIG_FRAMES: trig=True
                        if trig and not st["triggered"]:
                            st.update(triggered=True,trigger_time=frame_ts,frames_in_trigger=0)
                        if st["triggered"]:
                            st["frames_in_trigger"]+=1
                            if fr.get("recovered_quickly") or abs(avn)>mth*4:
                                st.update(triggered=False,trigger_time=None,frames_in_trigger=0)
                            elif st["trigger_time"] and (frame_ts-st["trigger_time"])>CONF_WIN:
                                st.update(triggered=False,trigger_time=None,frames_in_trigger=0)
                    except Exception: pass

                    # ── Hand SOS (Zone-gated) ────────────────────────────
                    now_t = frame_ts  # [QA-P0-4] frame time, not processing time
                    prev_c_info = track_prev_center.get(tid)
                    track_prev_center[tid] = (cx, cy, now_t)
                    vel_px_s = 0.0
                    if prev_c_info:
                        dt = max(1e-3, now_t - prev_c_info[2])
                        vel_px_s = (((cx - prev_c_info[0])**2 + (cy - prev_c_info[1])**2)**0.5) / dt

                    both_overhead = _are_both_arms_over_head(kp)
                    is_lying = (
                        posture_class == "laying" or
                        fr.get("is_fallen") or
                        fr.get("danger_lying") or
                        (fr.get("spine_angle") is not None and float(fr.get("spine_angle")) < 40.0) or
                        (fr.get("bbox_ratio") is not None and float(fr.get("bbox_ratio")) > 1.2)
                    )
                    recently_fallen = (now_t - _fall_last_event_ts.get(tid, -1e9)) < 25.0
                    # Suppress if moving rapidly (running > 0.45 frame_h/s, not normal stepping/gesturing)
                    is_moving_fast = (vel_px_s / max(1.0, float(h))) > 0.45

                    # Suppress Hand SOS if disabled by signal_config, both arms overhead (Pose SOS), fallen, lying, or moving fast
                    if (not hand_active) or both_overhead or fall_ev.get(tid) or is_lying or recently_fallen or is_moving_fast:
                        hand_states[tid] = 0
                        hand_consec3[tid] = 0
                        hand_recent_sos[tid].clear()
                    else:
                        in_hand_zone = zones.in_zone(cx, cy, detector_type="hand_sos")
                        prev_hs = hand_states.get(tid, 0)
                        hs = prev_hs
                        hdet = False
                        if tid not in hand_first_seen:
                            hand_first_seen[tid] = now_t
                        x1, y1, x2, y2 = bbox
                        bbox_area_norm = (max(0.0, x2 - x1) * max(0.0, y2 - y1)) / max(1.0, float(w * h))
                        track_age_sec = now_t - hand_first_seen[tid]
                        hand_eligible = (
                            in_hand_zone and
                            bbox_area_norm >= hand_min_bbox_area_norm and
                            track_age_sec >= hand_min_track_age_seconds
                        )
                        try:
                            if hand_eligible:
                                hand_lms = hand_d.process_wrist_crop(frame_clahe, kp, h, w, bbox=bbox)
                                if hand_lms:
                                    hl = hand_lms[0]
                                    hdet = True
                                    hs = hand_d.check_sos_step(hs, hl)
                        except Exception: pass

                        # Step transition: allow up to 3.5s per step
                        if hs != prev_hs:
                            hand_step_time[tid] = now_t
                        elif hs >= 1:
                            if now_t - hand_step_time.get(tid, now_t) > 3.5:
                                hs = 0
                                hand_consec3[tid] = 0

                        if hdet:
                            hand_miss[tid] = 0
                        else:
                            hand_miss[tid] = hand_miss.get(tid, 0) + 1
                            if hand_miss[tid] >= GRACE_FRAMES:
                                hs = max(0, hs - 1)
                                hand_miss[tid] = 0
                        # [FIX-FP] นับ state=3 ต่อเนื่อง — ต้องเป็นท่ากำหมัดจริง (นิ้วปิด และไม่ใช่แบมือ)
                        is_fist_now = bool(hdet and hs == 3 and not hand_d._palm_open(hl) and hand_d._fingers_closed(hl))
                        if is_fist_now:
                            hand_consec3[tid] = hand_consec3.get(tid, 0) + 1
                        elif hdet:
                            # Only reset if hand is actually detected but is not a fist (e.g. open palm)
                            hand_consec3[tid] = 0
                        # If not hdet, keep hand_consec3 (grace for distance flicker)
                        consec_ok = hand_consec3.get(tid, 0) >= MIN_CONSEC3
                        is_sos_frame = bool(hs == 3 and hand_eligible and consec_ok)
                        recent_sos = hand_recent_sos[tid]
                        recent_sos.append(is_sos_frame)
                        sos_hits = sum(1 for ok in recent_sos if ok)
                        temporal_confirmed = (
                            len(recent_sos) >= hand_temporal_hits_required
                            and sos_hits >= hand_temporal_hits_required
                        )
                        sos_confirmed = temporal_confirmed

                        if sos_confirmed and not fall_ev.get(tid) and not is_lying and not recently_fallen:
                            can_emit = (now_t - _hand_last_event_ts.get(tid, -1e9)) >= hand_cooldown_seconds
                            if can_emit and not hand_ev.get(tid):
                                # Generate Base64 snapshot and save local image/JSON metadata/log
                                b64_snap, snap_file = save_annotated_snapshot(
                                    clean_frame,
                                    event_type="hand_sos",
                                    bbox=bbox,
                                    label=f"HAND SOS [tid={tid}]",
                                    color=(0, 255, 0),
                                    coord_size=(w, h),
                                    snapshot_dir=snapshot_dir,
                                    save_local=save_local_snapshots,
                                    extra_meta={
                                        "source_id": source_id,
                                        "track_id": tid,
                                        "confidence": conf,
                                        "location": location,
                                    },
                                    all_tracks=all_frame_tracks,
                                )
                                raw_b64 = b64_snap.split(",", 1)[1] if "," in b64_snap else b64_snap
                                try:
                                    ev_uuid = str(uuid.uuid4())
                                    insert_incident(
                                        event_uuid     = ev_uuid,
                                        detection_type = "Gesture",
                                        zone           = location,
                                        confidence     = conf,
                                        timestamp      = datetime.utcnow().isoformat(),
                                        severity       = "High",
                                        metadata       = {
                                            "source_id":       source_id,
                                            "track_id":        tid,
                                            "snapshotUrl":     b64_snap,
                                            "snapshot_base64": raw_b64,
                                            "imagePath":       snap_file or b64_snap,
                                            "snapshot_file":   snap_file,
                                            "gesture":         "hand_sos",
                                            "bbox":            [int(v) for v in bbox],
                                            "bbox_norm":       [round(bbox[0]/w, 4), round(bbox[1]/h, 4), round(bbox[2]/w, 4), round(bbox[3]/h, 4)],
                                            "frame_size":      [w, h],
                                        },
                                    )
                                    print(f"  [DB INSERTED] Hand SOS uuid={ev_uuid} | file={snap_file or 'base64'}")
                                except Exception as e:
                                    print(f"[DB] hand_sos insert error: {e}")
                                _hand_last_event_ts[tid] = now_t
                                hand_states[tid] = 0
                                hand_consec3[tid] = 0
                                hand_recent_sos[tid].clear()
                                hand_ev[tid] = True
                        else:
                            if not sos_confirmed:
                                hand_ev[tid] = False

                    # ── Hands Over Head SOS (Pose SOS) ───────────────────
                    in_pose_zone = zones.in_zone(cx, cy, detector_type="pose_sos")
                    if pose_active and in_pose_zone and not fall_ev.get(tid) and not (fr.get("is_fallen") or fr.get("danger_lying")):
                        head_res = head_d.process(tid, kp, h, w, timestamp=now_t)
                        can_emit_pose = (now_t - _pose_last_event_ts.get(tid, -1e9)) >= pose_cooldown_seconds
                        is_spatial_pose_dup = spatial_dedup.should_suppress("pose_sos", cx, cy, now_t, cooldown_sec=pose_cooldown_seconds)
                        if head_res.get("triggered") and can_emit_pose and not is_spatial_pose_dup:
                            b64_snap, snap_file = save_annotated_snapshot(
                                clean_frame,
                                event_type="pose_sos",
                                bbox=bbox,
                                label=f"POSE SOS [tid={tid}] held={head_res.get('time_held', 0.0):.1f}s",
                                color=(0, 165, 255),
                                coord_size=(w, h),
                                snapshot_dir=snapshot_dir,
                                save_local=save_local_snapshots,
                                extra_meta={
                                    "source_id": source_id,
                                    "track_id": tid,
                                    "confidence": conf,
                                    "location": location,
                                    "time_held": head_res.get("time_held", 0.0),
                                },
                                all_tracks=all_frame_tracks,
                            )
                            raw_b64 = b64_snap.split(",", 1)[1] if "," in b64_snap else b64_snap
                            try:
                                ev_uuid = str(uuid.uuid4())
                                insert_incident(
                                    event_uuid     = ev_uuid,
                                    detection_type = "Gesture",
                                    zone           = location,
                                    confidence     = conf,
                                    timestamp      = datetime.utcnow().isoformat(),
                                    severity       = "High",
                                    metadata       = {
                                        "source_id":       source_id,
                                        "track_id":        tid,
                                        "snapshotUrl":     b64_snap,
                                        "snapshot_base64": raw_b64,
                                        "imagePath":       snap_file or b64_snap,
                                        "snapshot_file":   snap_file,
                                        "gesture":         "pose_sos",
                                        "time_held":       head_res.get("time_held", 0.0),
                                        "bbox":            [int(v) for v in bbox],
                                        "bbox_norm":       [round(bbox[0]/w, 4), round(bbox[1]/h, 4), round(bbox[2]/w, 4), round(bbox[3]/h, 4)],
                                        "frame_size":      [w, h],
                                    },
                                )
                                print(f"  [DB INSERTED] Pose SOS uuid={ev_uuid} | file={snap_file or 'base64'}")
                            except Exception as e:
                                print(f"[DB] pose_sos insert error: {e}")
                            _pose_last_event_ts[tid] = now_t

                    # ── Fall CSV ─────────────────────────────────────────
                    if not fr.get("recovered_quickly"):
                        # [FALL-DEBUG] person found but fall not confirmed
                        if not fr.get("is_fallen"):
                            logger.debug(
                                "[FALL-DEBUG] frame %d (%.1fs) tid=%d: "
                                "person detected but fall NOT confirmed — "
                                "ratio=%.2f angle=%.1f is_down=%s ground_time=%s "
                                "model_lying=%s geom_time=%.1f used_stale_angle=%s",
                                n, n / max(1, SAMPLING_FPS), tid,
                                fr.get("bbox_ratio", 0), fr.get("spine_angle", 0),
                                fr.get("is_down"), fr.get("ground_time"),
                                fr.get("model_confident_lying"), fr.get("geometry_time", 0),
                                fr.get("used_stale_angle"),
                            )
                        if FALL_DEBUG_CSV:
                          try:
                            from pipeline import LOG_DIR
                            lp=LOG_DIR/"fall_vels.csv"
                            if not lp.exists():
                                lp.write_text("ts,tid,vel_y_px_s,vel_y_norm,avg_vel_px_s,avg_vel_norm,"
                                              "is_down,is_fallen,is_critical,spike_time,ground_time,"
                                              "time_to_ground,time_lying,danger_lying\n")
                            with open(lp,"a") as lf:
                                lf.write(f"{now_time},{tid},{fr.get('vel_y',0)},{fr.get('vel_y_norm',0)},"
                                         f"{fr.get('avg_vel',0)},{fr.get('avg_vel_norm',0)},"
                                         f"{int(fr.get('is_down',0))},{int(fr.get('is_fallen',0))},"
                                         f"{int(fr.get('is_critical',0))},{fr.get('spike_time')},"
                                         f"{fr.get('ground_time')},{fr.get('time_to_ground')},"
                                         f"{fr.get('time_lying')},{int(fr.get('danger_lying',0))}\n")
                          except Exception: pass

                    # ── Critical Fall & Confirmed Fall ────────────────────
                    esc=fr.get("danger_lying") and not fr.get("recovered_quickly")
                    is_critical=fr.get("is_critical") and not fr.get("recovered_quickly")
                    is_confirmed=(fr.get("is_fallen") or esc) and not fr.get("recovered_quickly")

                    if fall_active and (is_critical or is_confirmed):
                        # ── Flapping dedup (hysteresis) ──
                        # If this tid had a fall alert recently (within
                        # confirm_seconds * hysteresis_factor), treat it as
                        # the SAME event — don't dispatch a new alert.
                        prev_ts = _fall_last_event_ts.get(tid)
                        dedup_window = max(FALL_CONFIRM_S * FALL_HYSTERESIS_FACTOR, FALL_COOLDOWN_S)
                        is_flap = bool(
                            prev_ts is not None
                            and (now_t - prev_ts) < dedup_window
                        )
                        is_spatial_dup = spatial_dedup.should_suppress("fall", cx, cy, now_t, cooldown_sec=dedup_window)

                        if is_flap or is_spatial_dup:
                            logger.debug(
                                "[FALL-DEDUP] tid=%d: suppressed flapping/spatial duplicate alert "
                                "(is_flap=%s, is_spatial=%s)",
                                tid, is_flap, is_spatial_dup,
                            )
                            # Keep fall_ev[tid]=True so we don't re-trigger
                            fall_ev[tid] = True
                        elif not fall_ev.get(tid):
                            fall_bc[tid] = conf; fall_bt[tid] = now_time
                            ex = {
                                "track_id": tid, "source": source_id, "location": location,
                                "recovered_quickly": fr.get("recovered_quickly"), "fall_result": fr
                            }
                            if is_critical: ex["critical"] = True
                            if esc: ex["auto_escalated_immobile"] = True
                            lv, ln, flags = disp._assess_alert_level("fall", ex)

                            b64_snap, snap_file = save_annotated_snapshot(
                                clean_frame,
                                event_type="fall",
                                bbox=bbox,
                                label=f"FALL [tid={tid}] {fr.get('spine_angle', 0.0):.0f}deg",
                                color=(0, 0, 255),
                                coord_size=(w, h),
                                snapshot_dir=snapshot_dir,
                                save_local=save_local_snapshots,
                                extra_meta={
                                    **ex,
                                    "severity_name": ln,
                                    "confidence": conf,
                                },
                                all_tracks=all_frame_tracks,
                            )
                            raw_b64 = b64_snap.split(",", 1)[1] if "," in b64_snap else b64_snap
                            try:
                                ev_uuid = str(uuid.uuid4())
                                insert_incident(
                                    event_uuid     = ev_uuid,
                                    detection_type = "Fall",
                                    zone           = location,
                                    confidence     = conf,
                                    timestamp      = now_time,
                                    severity       = lv,
                                    metadata       = {
                                        **ex,
                                        "severity_name":   ln,
                                        "source_id":       source_id,
                                        "track_id":        tid,
                                        "snapshotUrl":     b64_snap,
                                        "snapshot_base64": raw_b64,
                                        "imagePath":       snap_file or b64_snap,
                                        "snapshot_file":   snap_file,
                                        "bbox":            [int(v) for v in bbox],
                                        "bbox_norm":       [round(bbox[0]/w, 4), round(bbox[1]/h, 4), round(bbox[2]/w, 4), round(bbox[3]/h, 4)],
                                        "frame_size":      [w, h],
                                    }
                                )
                                print(f"  [DB INSERTED] Fall uuid={ev_uuid} | file={snap_file or 'base64'}")
                            except Exception as e:
                                print(f"[DB] fall insert error: {e}")
                            _fall_last_event_ts[tid] = now_t
                            fall_ev[tid] = True
                    else:
                        if fall_ev.get(tid):
                            fall_ev[tid]=False;fall_bc[tid]=0;fall_bf[tid]=None;fall_bt[tid]=None

            zones.draw(frame)  # [QA-P0-3] overlay only after detection
            viz.fps(frame)


    except KeyboardInterrupt:
        print(f"\n[{source_id[-30:]}] Stopped.")
    finally:
        if cap is not None: cap.release()
        if reader is not None: reader.release()
        hand_d.release()
        if vlc_mgr: vlc_mgr.stop()
        print(f"[{source_id[-30:]}] Done.")

def run_source(s):
    try:
        silence_stream_noise()
        cid = s.get("id") or s.get("code") or ""
        main(
            s["path"],
            port=s.get("port", 8081),
            location=s.get("location", ""),
            cam_id=cid,
            use_vlc=s.get("use_vlc", False),
            loop=s.get("loop", False),
            signal_config=s.get("signal_config"),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[run_source] {s.get('id')} error: {e}")

if __name__=="__main__":
    try: multiprocessing.set_start_method("spawn",force=True)
    except RuntimeError: pass

    import argparse
    parser = argparse.ArgumentParser(description="Multi-Source SOS Detection System")
    parser.add_argument("--source", "-s", type=str, help="Single RTSP stream URL or video file path")
    parser.add_argument("--dir", "-d", "-dir", dest="dir", type=str, help="Directory containing video files")
    parser.add_argument("--location", "-l", type=str, default="", help="Location name")
    parser.add_argument("--cam-id", "-c", type=str, default="", help="Camera ID")
    parser.add_argument("--port", "-p", type=int, default=8081, help="Streaming port (base)")
    parser.add_argument("--use-vlc", action="store_true", help="Force transcoding via VLC HTTP stream")
    parser.add_argument("--loop", action="store_true", help="Loop video file continuously")
    parser.add_argument("--from-db", action="store_true", help="Load active cameras dynamically from MongoDB Atlas (cctv_cameras)")
    args, unknown = parser.parse_known_args()

    sources = []
    if args.from_db:
        target_cam = args.cam_id if args.cam_id else None
        target_msg = f"for cam_id={target_cam}" if target_cam else "(all active)"
        print(f"[main] Fetching cameras from MongoDB Atlas {target_msg}...")
        db_cams = db_module.get_active_cameras(cam_id=target_cam)
        for i, c in enumerate(db_cams):
            if not c.get("path"):
                print(f"⚠️  Camera {c.get('code')} has no RTSP URL or could not be decrypted — skipping")
                continue
            sources.append({
                "id": c.get("code") or f"cam_{i+1}",
                "path": c["path"],
                "location": c.get("location") or c.get("name", ""),
                "port": args.port + i,
                "use_vlc": args.use_vlc,
                "loop": False,
                "signal_config": c.get("signal_config"),
                "enabled": True,
            })
        print(f"[main] Loaded {len(sources)} active camera(s) from database.")
    elif args.source:
        p = Path(args.source)
        loc = args.location or (p.stem if p.exists() else "stream")
        target_cam = args.cam_id or ""
        sig_cfg = db_module.get_camera_signal_config(target_cam) if (target_cam and db_module) else None
        sources.append({
            "id": args.cam_id or "source_1",
            "path": args.source,
            "location": loc,
            "port": args.port,
            "use_vlc": args.use_vlc,
            "loop": args.loop,
            "signal_config": sig_cfg,
            "enabled": True,
        })
    elif args.dir:
        p = Path(args.dir)
        if not p.exists():
            print(f"❌ ไม่พบโฟลเดอร์: {args.dir}")
            sys.exit(1)
        vids = sorted([f for f in p.iterdir() if f.suffix.lower() in [".mp4", ".avi", ".mkv", ".m4v", ".mov"]])
        if not vids:
            print(f"⚠️ ไม่พบไฟล์วิดีโอในโฟลเดอร์: {args.dir}")
            sys.exit(1)
        for i, v in enumerate(vids):
            sources.append({
                "id": f"cam_{i+1}",
                "path": str(v),
                "location": v.stem,
                "port": args.port + i,
                "use_vlc": args.use_vlc,
                "loop": args.loop,
                "enabled": True,
            })
    elif (ROOT / "config/sources.yaml").exists():
        try:
            all_sources = yaml.safe_load((ROOT / "config/sources.yaml").read_text()).get("sources", [])
            sources = [s for s in all_sources if s.get("enabled", True)]
        except Exception as e:
            print(f"⚠️ อ่าน config/sources.yaml ผิดพลาด: {e}")
    else:
        print("❌ ไม่พบ config/sources.yaml และไม่มีการระบุ --source, --dir หรือ --from-db")
        print("\nตัวอย่างการใช้งาน:")
        print("  python3 main.py --from-db")
        print("  python3 main.py --source 'rtsp://...'")
        print("  python3 main.py --source '/path/to/video.mp4'")
        print("  python3 main.py --dir '/path/to/videos_folder'")
        sys.exit(1)

    if not sources:
        print("⚠️ ไม่มี Source สำหรับประมวลผล — ออกจากการทำงาน")
        sys.exit(0)

    print(f"\n{'='*60}")
    print(f"Multi-Source Detection System")
    print(f"{'='*60}")
    print(f"Enabled: {len(sources)} sources")
    for s in sources:
        print(f"  ✓ {s.get('id','?'):20} | {s.get('location','?'):15} | port {s.get('port',8081)}")
    print(f"{'='*60}\n")
    procs=[]
    for s in sources:
        p=multiprocessing.Process(target=run_source,args=(s,))
        p.daemon=False; p.start(); procs.append((s,p))
    try:
        while True:
            if not any(p.is_alive() for _,p in procs): print("All done"); break
            time.sleep(5)
    except KeyboardInterrupt: print("\nStopping …")
    finally:
        for _,p in procs:
            if p.is_alive(): p.terminate()
        print("Main exiting.")
