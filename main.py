import cv2, sys, time, yaml, torch, subprocess, platform, socket, requests, io
from pathlib import Path
from collections import defaultdict, deque
import numpy as np

# MUST BE FIRST: Initialize environment before other imports
from config.env_manager import init_env
if not init_env(require_edit=False):
    print("❌ Environment initialization failed. Exiting.")
    sys.exit(1)

from ultralytics import YOLO
from detectors.fall_detector import FallDetector
from detectors.hand_sos_detector import HandSOSDetector
from pipeline  import CooldownEngine, ZoneManager, AlertDispatcher
from utils     import preprocess, Visualizer
from help_request_dispatcher import HelpRequestDispatcher
import database

cfg  = yaml.safe_load(Path("config/thresholds.yaml").read_text())
# streaming config (lightweight trigger)
streaming_cfg = cfg.get('streaming', {})
SAMPLING_FPS = streaming_cfg.get('fps', 10)
ROLLING_WINDOW_FRAMES = streaming_cfg.get('rolling_window', 90)
CONFIRMATION_WINDOW = streaming_cfg.get('confirmation_window', 30)
TRIGGER_ANGLE_DELTA = streaming_cfg.get('trigger_angle_delta', 30)
TRIGGER_FRAMES = streaming_cfg.get('trigger_frames', 5)
LYING_ANGLE_THRESH = streaming_cfg.get('lying_angle_thresh', 45)

W, H = 1920, 1080
SKIP = cfg.get("general",{}).get("frame_skip", 2)
CONF = cfg.get("general",{}).get("person_conf", 0.60)

device = ("mps"  if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def start_vlc(src, port=8081):
    if src.startswith("http"):
        return None, src
    if port == 8081:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
    vlc = ("/Applications/VLC.app/Contents/MacOS/VLC"
           if platform.system()=="Darwin" else "cvlc")
    p = subprocess.Popen([
        vlc, src,
        f"--sout=#transcode{{vcodec=MJPG,vb=3000,vfilter=canvas{{width={W},height={H}}},fps={SAMPLING_FPS}}}"
        f":http{{mux=mpjpeg,dst=:{port}/}}",
        "--sout-keep","--loop",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"VLC PID={p.pid} — waiting 3s...")
    time.sleep(3)
    if p.poll() is not None:
        raise RuntimeError(f"VLC exited while starting the local stream for: {src}")
    print(f"VLC stream ready at http://127.0.0.1:{port}/")
    return p, f"http://127.0.0.1:{port}/"

# Detector server URL (model-server)
DETECTOR_URL = cfg.get('general',{}).get('detector_url','http://127.0.0.1:9000')

# Simple local tracker to assign persistent IDs when server returns per-frame detections
class ItemWrap:
    def __init__(self,val):
        self.val = val
    def cpu(self):
        return self
    def numpy(self):
        return np.array(self.val)
    def item(self):
        return self.numpy().item()
    def __float__(self):
        return float(self.item())
    def __int__(self):
        return int(self.item())

class BoxesWrapper:
    def __init__(self, xyxy, confs, ids):
        self.xyxy = [ItemWrap(x) for x in xyxy]
        self.conf = [ItemWrap(c) for c in confs]
        self.id = [ItemWrap(i) for i in ids] if ids is not None else None
    def __len__(self):
        return len(self.xyxy)

class KeypointsWrapper:
    def __init__(self, data):
        self.data = [ItemWrap(d) for d in data] if data else None
    def __len__(self):
        return len(self.data or [])

class SimpleTracker:
    def __init__(self, iou_thresh=0.3, max_lost=5):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.next_id = 0
        self.tracks = {}  # id -> {'bbox': [x1,y1,x2,y2], 'lost': int}
    def _iou(self, a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        w = max(0, x2 - x1); h = max(0, y2 - y1)
        inter = w * h
        area_a = max(1e-6, (a[2]-a[0])*(a[3]-a[1]))
        area_b = max(1e-6, (b[2]-b[0])*(b[3]-b[1]))
        union = area_a + area_b - inter
        return inter / union if union>0 else 0.0
    def update(self, dets):
        # dets: list of dicts with 'bbox'
        assigned_ids = []
        for det in dets:
            best_id = None; best_iou = 0.0
            for tid, t in self.tracks.items():
                iou = self._iou(det['bbox'], t['bbox'])
                if iou > best_iou:
                    best_iou = iou; best_id = tid
            if best_iou >= self.iou_thresh and best_id not in assigned_ids:
                det['track_id'] = best_id
                self.tracks[best_id]['bbox'] = det['bbox']
                self.tracks[best_id]['lost'] = 0
                assigned_ids.append(best_id)
            else:
                tid = self.next_id; self.next_id += 1
                det['track_id'] = tid
                self.tracks[tid] = {'bbox': det['bbox'], 'lost': 0}
                assigned_ids.append(tid)
        # increment lost and cleanup
        current_ids = set(d['track_id'] for d in dets)
        for tid in list(self.tracks.keys()):
            if tid not in current_ids:
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost:
                    del self.tracks[tid]
        return dets

tracker = SimpleTracker()

fall_d   = FallDetector(cfg.get("fall",{}))
pose_d = None
hand_d   = HandSOSDetector(cfg.get("hand_sos",{}))


def detect_frame(frame):
    # encode to jpeg and POST to model-server
    _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    try:
        r = requests.post(DETECTOR_URL + '/detect', files={'image': ('frame.jpg', buf.tobytes(), 'image/jpeg')}, timeout=5)
        if r.status_code != 200:
            return []
        return r.json().get('detections', [])
    except Exception:
        return []

zones    = ZoneManager("config/zones.yaml","default")
# AlertDispatcher cooldowns: read per-event cooldowns from config (fall/hand_sos) with a default
alert_default = cfg.get("general",{}).get("alert_cooldown_seconds", cfg.get("fall",{}).get("cooldown_seconds", 120))
alert_cds = {
    "fall": cfg.get("fall",{}).get("cooldown_seconds", alert_default),
    "hand_sos": cfg.get("hand_sos",{}).get("cooldown_seconds", alert_default),
}
# enforce single alert per local file (config.general.one_alert_per_file, default True)
enforce_one = cfg.get('general',{}).get('one_alert_per_file', True)

# Initialize help dispatcher
help_disp = HelpRequestDispatcher(
    webhook_url="",  # Loaded from .env via HelpRequestDispatcher.__init__
    timeout=5,
    db=database
)

disp     = AlertDispatcher(cooldowns=alert_cds, default_cooldown=alert_default, enforce_one_per_file=enforce_one, help_dispatcher=help_disp, db=database)
viz      = Visualizer()
cds      = defaultdict(lambda:{
    "fall_warning": CooldownEngine("fall_warning", cfg.get("fall_warning", cfg.get("fall",{}))),
    "fall":     CooldownEngine("fall",     cfg.get("fall",{})),
})
hand_cd  = CooldownEngine("hand_sos", cfg.get("hand_sos",{}))

def main(src, port=8081, location=None):
    # normalize source identifier to absolute path when possible so each file/source is unique
    try:
        source_id = str(Path(src).resolve()) if src and Path(src).exists() else str(src)
    except Exception:
        source_id = str(src)
    # optional human-friendly location string from sources.yaml
    source_location = location or None

    vlc_proc, url = start_vlc(src, port)

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Cannot open: {url}"); return

    print(f"Connected: {url}")
    print("Headless mode — alerts/ & logs/alerts.log")
    print("Press Ctrl+C to stop\n")

    n = 0
    try:
        # State for unique event logging (per-track for hand SOS)
        hand_event_active = dict()
        hand_best_conf = dict()
        hand_best_frame = dict()
        hand_best_time = dict()
        hand_states = dict()  # per-track state 0..3

        fall_event_active = dict()
        fall_best_conf = dict()
        fall_best_frame = dict()
        fall_best_time = dict()

        # Streaming/lightweight trigger buffers and per-track streaming state
        frame_buffers = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW_FRAMES))  # raw frames per source
        streaming_states = defaultdict(lambda: {
            "angle_hist": deque(maxlen=TRIGGER_FRAMES),
            "motion_hist": deque(maxlen=TRIGGER_FRAMES),
            "triggered": False,
            "trigger_time": None,
            "frames_in_trigger": 0,
        })

        running = True
        while running:
            ret, raw = cap.read()
            if not ret:
                time.sleep(0.05); continue
            n += 1
            if n % SKIP != 0:
                continue

            frame = preprocess(raw, W, H)
            h, w  = frame.shape[:2]
            zones.draw(frame)

            # run hand detector to populate landmarks (association done per-track later)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                hand_d.process_frame(rgb)
            except Exception:
                pass

            now_time = time.strftime("%Y-%m-%d %H:%M:%S")

            # fall + pose (per track id)
            raw_dets = detect_frame(frame)
            if raw_dets:
                # assign tracks
                assigned = tracker.update(raw_dets)
                # build wrapper objects expected by downstream code
                xyxy = [d['bbox'] for d in assigned]
                confs = [d.get('conf', 0.0) for d in assigned]
                ids = [d.get('track_id') for d in assigned]
                boxes = BoxesWrapper(xyxy, confs, ids)
                kpts = KeypointsWrapper([d.get('keypoints', []) for d in assigned])
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    x1, y1, x2, y2 = bbox
                    conf = float(boxes.conf[i].cpu())
                    tid  = int(boxes.id[i].cpu()) if boxes.id is not None else i
                    if kpts is None or not kpts.data or i>=len(kpts.data): continue
                    kp = kpts.data[i].cpu().numpy()
                    cx = (bbox[0]+bbox[2])/2/w
                    cy = (bbox[1]+bbox[3])/2/h
                    if not zones.in_zone(cx,cy): continue

                    fr = fall_d.process(tid,kp,bbox,h,w)
                    pr = {"is_sos": False}
                    cd = cds[tid]
                    now_time = time.strftime("%Y-%m-%d %H:%M:%S")

                    # Fall warning: dispatch a warning-level alert earlier using fall_warning cooldown
                    try:
                        fw_cd = cd.get("fall_warning") if isinstance(cd, dict) else None
                        is_down = bool(fr.get("is_down") and not fr.get("recovered_quickly"))
                        if fw_cd and fw_cd.update(is_down):
                            warn_frame = raw.copy()
                            label_w = f"FALL WARNING {now_time} @ {source_location or 'unknown'}"
                            cv2.putText(warn_frame, label_w, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,165,255), 2)
                            disp.dispatch("fall_warning", warn_frame, {"track_id":tid, "source": source_id, "location": source_location, "warning": True, "recovered_quickly": fr.get('recovered_quickly')})
                    except Exception:
                        pass

                    # --- Lightweight streaming trigger: maintain short history and detect quick triggers ---
                    try:
                        # keep raw frame history per source (for context/capture on trigger)
                        try:
                            frame_buffers[source_id].append(raw.copy())
                        except Exception:
                            pass

                        st = streaming_states[tid]
                        angle = float(fr.get('spine_angle') or 0.0)
                        avg_vel_norm = float(fr.get('avg_vel_norm') or fr.get('avg_vel_norm', 0.0) or 0.0)
                        st['angle_hist'].append(angle)
                        st['motion_hist'].append(abs(avg_vel_norm))

                        # simple triggers:
                        triggered = False
                        # A: sudden angle change
                        if len(st['angle_hist']) >= TRIGGER_FRAMES:
                            if (max(st['angle_hist']) - min(st['angle_hist'])) >= TRIGGER_ANGLE_DELTA:
                                triggered = True
                        # B: high motion then low motion with torso angle fairly large
                        motion_thresh = cfg.get('fall',{}).get('motion_thresh_norm', 0.02)
                        if len(st['motion_hist']) >= 2:
                            if st['motion_hist'][-2] > motion_thresh*3 and st['motion_hist'][-1] < motion_thresh and angle >= LYING_ANGLE_THRESH:
                                triggered = True
                        # C: sustained high torso angle
                        if sum(1 for a in st['angle_hist'] if a >= LYING_ANGLE_THRESH) >= TRIGGER_FRAMES:
                            triggered = True

                        # if newly triggered, set timestamp and emit a light warning
                        if triggered and not st['triggered']:
                            st['triggered'] = True
                            st['trigger_time'] = time.time()
                            st['frames_in_trigger'] = 0
                            try:
                                warn_frame = raw.copy()
                                label_w = f"LIGHT FALL TRIGGER {now_time} @ {source_location or 'unknown'}"
                                cv2.putText(warn_frame, label_w, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,165,255), 2)
                                # lightweight trigger uses same 'fall_warning' event type so existing cooldowns apply
                                disp.dispatch("fall_warning", warn_frame, {"track_id":tid, "source": source_id, "location": source_location, "trigger": "light", "recovered_quickly": fr.get('recovered_quickly'), "fall_result": fr})
                            except Exception:
                                pass

                        # if already triggered, update frames_in_trigger and check for recovery
                        if st['triggered']:
                            st['frames_in_trigger'] += 1
                            # quick recovery clears trigger
                            if fr.get('recovered_quickly') or abs(avg_vel_norm) > motion_thresh*4:
                                st['triggered'] = False
                                st['trigger_time'] = None
                                st['frames_in_trigger'] = 0
                            else:
                                # if trigger persists longer than confirmation window, keep waiting for full detector to confirm
                                if st['trigger_time'] and (time.time() - st['trigger_time']) > CONFIRMATION_WINDOW:
                                    # give console feedback and continue — fall_d.process will perform the conservative confirmation
                                    print(f"Trigger window expired for track {tid} (source={source_id}) — awaiting full confirmation")
                                    # clear trigger to avoid repeated messages; keep waiting for is_fallen to be set by fall detector
                                    st['triggered'] = False
                                    st['trigger_time'] = None
                                    st['frames_in_trigger'] = 0
                    except Exception:
                        pass

                    # Hand SOS: associate hand landmarks to this track and run per-track state machine
                    hand_state = hand_states.get(tid, 0)
                    hand_detected_in_bbox = False
                    try:
                        results_h = getattr(hand_d, "_results", None)
                        if results_h and getattr(results_h, "hand_landmarks", None):
                            for hand_landmarks in results_h.hand_landmarks:
                                # compute bbox and center for hand
                                xs = [lm.x for lm in hand_landmarks]
                                ys = [lm.y for lm in hand_landmarks]
                                minx, maxx = min(xs), max(xs)
                                miny, maxy = min(ys), max(ys)
                                # normalized bbox area
                                box_w = (maxx - minx) * w
                                box_h = (maxy - miny) * h
                                area_norm = (box_w * box_h) / (w * h)
                                # keypoint confidences if available
                                kp_conf_ok = True
                                try:
                                    kp_conf_ok = all(getattr(lm, 'visibility', 1.0) >= cfg.get('hand_sos',{}).get('min_kp_conf', 0.4) for lm in hand_landmarks)
                                except Exception:
                                    pass
                                # skip small/low-confidence hands
                                if area_norm < cfg.get('hand_sos',{}).get('min_hand_bbox_area_norm', 0.002) or not kp_conf_ok:
                                    continue
                                pts = [(int(lm.x*w), int(lm.y*h)) for lm in hand_landmarks]
                                hx = sum(p[0] for p in pts)/len(pts)
                                hy = sum(p[1] for p in pts)/len(pts)
                                if x1 <= hx <= x2 and y1 <= hy <= y2:
                                    hand_detected_in_bbox = True
                                    # update state machine using detector helpers
                                    try:
                                        if hand_state==0 and hand_d._palm_open(hand_landmarks):
                                            hand_state = 1
                                        elif hand_state==1 and hand_d._thumb_in(hand_landmarks):
                                            hand_state = 2
                                        elif hand_state==2 and hand_d._fingers_closed(hand_landmarks):
                                            hand_state = 3
                                    except Exception:
                                        pass
                                    break
                    except Exception:
                        pass

                    if not hand_detected_in_bbox:
                        hand_state = max(0, hand_state-1)
                    hand_states[tid] = hand_state

                    # handle hand event per-track (only if this track not fallen)
                    if hand_state==3 and not fall_event_active.get(tid):
                        if not hand_event_active.get(tid) or conf > hand_best_conf.get(tid, 0):
                            hand_best_conf[tid] = conf
                            hand_best_frame[tid] = raw.copy()
                            hand_best_time[tid] = now_time
                        hand_event_active[tid] = True
                    elif hand_event_active.get(tid):
                        if hand_best_frame.get(tid) is not None:
                            label = f"SOS HAND {hand_best_time[tid]} @ {source_location or 'unknown'}"
                            cv2.putText(hand_best_frame[tid], label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                            extra_h = {"track_id":tid, "source": source_id, "location": source_location}
                            saved = disp.dispatch("hand_sos", hand_best_frame[tid], extra_h)
                            # clear per-track bests so we don't immediately re-dispatch
                            hand_event_active[tid] = False
                            hand_best_conf[tid] = 0
                            hand_best_frame[tid] = None
                            hand_best_time[tid] = None
                            # saved alert for this (source,atype); do not stop processing — allow other event types to be captured
                            pass

                    # log velocities for tuning (append CSV) — skip if recovered_quickly (stumble)
                    try:
                        # If this event was a quick recovery (stumble), do not record it
                        if not fr.get('recovered_quickly'):
                            from pipeline import LOG_DIR
                            logp = LOG_DIR/"fall_vels.csv"
                            if not logp.exists():
                                logp.write_text("ts,tid,vel_y_px_s,vel_y_norm,avg_vel_px_s,avg_vel_norm,is_down,is_fallen,is_critical,spike_time,ground_time,time_to_ground,time_lying,danger_lying\n")
                            with open(logp, "a") as lf:
                                lf.write(f"{now_time},{tid},{fr.get('vel_y',0)},{fr.get('vel_y_norm',0)},{fr.get('avg_vel',0)},{fr.get('avg_vel_norm',0)},{int(fr.get('is_down'))},{int(fr.get('is_fallen'))},{int(fr.get('is_critical'))},{fr.get('spike_time')},{fr.get('ground_time')},{fr.get('time_to_ground')},{fr.get('time_lying')},{int(fr.get('danger_lying'))}\n")
                    except Exception:
                        pass

                    # Immediate escalation for critical falls (no recovery within critical_seconds)
                    # Do not escalate or record if this was a quick recovery (stumble)
                    if fr.get("is_critical") and not fr.get('recovered_quickly'):
                        # keep best frame for this critical event
                        if not fall_best_conf.get(tid) or conf > fall_best_conf.get(tid, 0):
                            fall_best_conf[tid] = conf
                            fall_best_frame[tid] = raw.copy()
                            fall_best_time[tid] = now_time
                        # annotate and dispatch immediately with critical flag
                        label = f"FALL CRITICAL {fall_best_time[tid]} @ {source_location or 'unknown'}"
                        cv2.putText(fall_best_frame[tid], label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                        extra_c = {"track_id":tid, "critical":True, "source": source_id, "location": source_location, "recovered_quickly": fr.get('recovered_quickly'), "fall_result": fr}
                        saved = disp.dispatch("fall", fall_best_frame[tid], extra_c)
                        # mark active so we don't re-dispatch repeatedly
                        fall_event_active[tid] = True
                        # saved alert for this (source,atype); do not stop processing
                        pass

                    # FALL EVENT: dispatch immediately when is_fallen becomes true (conservative detection)
                    # Also escalate if subject has been lying immobile longer than danger.immobile_seconds
                    # Ignore and do not record if this was a quick recovery (stumble)
                    try:
                        immobile_thresh = cfg.get('danger',{}).get('immobile_seconds', 5)
                    except Exception:
                        immobile_thresh = 5

                    time_lying = fr.get('time_lying') or 0
                    # If lying longer than immobile threshold and not recovered_quickly, escalate even if is_fallen not yet True
                    # Only auto-escalate when detector marks danger_lying (sustained extended inactivity)
                    should_escalate_immobile = (fr.get('danger_lying') and not fr.get('recovered_quickly'))

                    if (fr.get("is_fallen") or should_escalate_immobile) and not fr.get('recovered_quickly'):
                        # first time we confirm fall for this tid -> dispatch alert
                        if not fall_event_active.get(tid):
                            fall_best_conf[tid] = conf
                            fall_best_frame[tid] = raw.copy()
                            fall_best_time[tid] = now_time
                            label = f"FALL {fall_best_time[tid]} @ {source_location or 'unknown'}"
                            cv2.putText(fall_best_frame[tid], label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                            extra = {"track_id":tid, "source": source_id, "location": source_location, "recovered_quickly": fr.get('recovered_quickly'), "fall_result": fr}
                            if should_escalate_immobile:
                                extra['auto_escalated_immobile'] = True
                            saved = disp.dispatch("fall", fall_best_frame[tid], extra)
                            # mark active so we don't re-dispatch repeatedly
                            fall_event_active[tid] = True
                            # saved alert for this (source,atype); do not stop processing
                            pass
                        else:
                            # update best frame if higher confidence later
                            if conf > fall_best_conf.get(tid, 0):
                                fall_best_conf[tid] = conf
                                fall_best_frame[tid] = raw.copy()
                                fall_best_time[tid] = now_time
                    else:
                        # recovered or not confirmed; clear active state
                        if fall_event_active.get(tid):
                            fall_event_active[tid] = False
                            fall_best_conf[tid] = 0
                            fall_best_frame[tid] = None
                            fall_best_time[tid] = None


            viz.fps(frame)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        hand_d.release()
        if vlc_proc:
            vlc_proc.terminate()
        print("Done. Check alerts/ and logs/")

import yaml
import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

def run_source(src):
    try:
        print(f"\n=== Running source: {src.get('id')} | {src.get('path')} | port {src.get('port')} ===")
        main(src.get('path'), src.get('port', 8081), src.get('location'))
    except Exception as e:
        print(f"Source {src.get('id')} raised exception: {e}")
    finally:
        print(f"Source {src.get('id')} finished")

if __name__=="__main__":
    # Load sources.yaml
    sources = yaml.safe_load(Path("config/sources.yaml").read_text())["sources"]
    procs = []
    for src in sources:
        p = multiprocessing.Process(target=run_source, args=(src,))
        p.daemon = False
        p.start()
        procs.append((src, p))

    # Monitor processes; don't block in a single join() so we can report statuses
    try:
        while True:
            alive = [ (s,p) for s,p in procs if p.is_alive() ]
            if not alive:
                print("All sources finished")
                break
            for s,p in procs:
                if p.is_alive():
                    print(f"Source {s.get('id')} (pid={p.pid}) running")
                else:
                    if p.exitcode is not None:
                        print(f"Source {s.get('id')} exited (code={p.exitcode})")
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: terminating child processes...")
        for s,p in procs:
            if p.is_alive():
                p.terminate()
        print("Terminated children")
    finally:
        for s,p in procs:
            if p.is_alive():
                p.terminate()
        print("Main exiting")
