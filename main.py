#!/usr/bin/env python3
"""
main_production.py — SOS + Object Guardian — Production v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features:
  • Fall detection  (FallDetector + streaming trigger)
  • Silent SOS hand  (HandSOSDetector state-machine)
  • Object left-behind / theft  (ObjectGuardian)
  • VLCStreamManager  (cross-platform, health-checked)
  • Structured JSONL logging per event
  • MongoDB (Atlas) real-time incident storage  ← NEW
"""
import cv2, sys, time, yaml, torch, requests, os   # ← os ย้ายขึ้นบนสุด
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime                        # ← เพิ่ม datetime
import numpy as np, multiprocessing, logging

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ──── Initialize environment (MUST BE FIRST) ────────────────────────────────
from config.env_manager import init_env
if not init_env(require_edit=False):
    print("❌ Environment initialization failed. Exiting.")
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

from detectors.fall_detector     import FallDetector
from detectors.hand_sos_detector import HandSOSDetector
from detectors.object_guardian   import ObjectGuardian
from pipeline                    import CooldownEngine, ZoneManager, AlertDispatcher
from help_request_dispatcher     import HelpRequestDispatcher
from utils                       import preprocess, Visualizer, add_sos_badge
from vlc_stream                  import VLCStreamManager
from alert_logger                import alert_logger
from database                    import _get_db, insert_incident  # ← NEW

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
DET_URL     = GEN.get("detector_url", "http://127.0.0.1:8000")
DEVICE      = ("mps" if torch.backends.mps.is_available() else
               "cuda" if torch.cuda.is_available() else "cpu")
print(f"[main] Device: {DEVICE}")


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

class SimpleTracker:
    def __init__(self): self.next_id=0; self.tracks={}
    def _iou(self,a,b):
        x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
        w=max(0,x2-x1);h=max(0,y2-y1);inter=w*h
        aa=max(1e-6,(a[2]-a[0])*(a[3]-a[1]));ab=max(1e-6,(b[2]-b[0])*(b[3]-b[1]))
        return inter/(aa+ab-inter) if (aa+ab-inter)>0 else 0.0
    def update(self,dets):
        used=[]
        for det in dets:
            best_id=None;best_iou=0.0
            for tid,t in self.tracks.items():
                iou=self._iou(det["bbox"],t["bbox"])
                if iou>best_iou: best_iou=iou;best_id=tid
            if best_iou>=0.3 and best_id not in used:
                det["track_id"]=best_id;self.tracks[best_id].update(bbox=det["bbox"],lost=0);used.append(best_id)
            else:
                tid=self.next_id;self.next_id+=1
                det["track_id"]=tid;self.tracks[tid]={"bbox":det["bbox"],"lost":0};used.append(tid)
        tid_set={d["track_id"] for d in dets}
        for tid in list(self.tracks):
            if tid not in tid_set:
                self.tracks[tid]["lost"]+=1
                if self.tracks[tid]["lost"]>5: del self.tracks[tid]
        return dets

def _api(endpoint, frame, timeout=5):
    _,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,80])
    try:
        r=requests.post(DET_URL+endpoint,files={"image":("f.jpg",buf.tobytes(),"image/jpeg")},timeout=timeout)
        if r.status_code==200: return r.json()
    except Exception: pass
    return {}

def main(src:str, port:int=8081, location:str=""):
    source_id=str(Path(src).resolve()) if Path(src).exists() else str(src)

    vlc_mgr=None
    if src.startswith("http") or src.startswith("rtsp"):
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

        # ── Health check: รอให้ HTTP stream พร้อมรับ connection ──────────
        print(f"[VLC] Waiting for stream at {url} ...")
        for attempt in range(10):
            if vlc_mgr.health_check():
                print(f"[VLC] Stream confirmed ready (attempt {attempt+1})")
                break
            time.sleep(1)
        else:
            print(f"⚠️  [VLC] Stream not responding after 10s — continuing anyway")

    cap=cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"[cap] Cannot open: {url}")
        if vlc_mgr: vlc_mgr.stop(); return

    print(f"[{source_id[-30:]}] Connected | location={location or '?'}")

    tracker =SimpleTracker()
    fall_d  =FallDetector(cfg.get("fall",{}))
    hand_d  =HandSOSDetector(cfg.get("hand_sos",{}))
    obj_grd =ObjectGuardian({**cfg.get("object_guardian",{}),"alert_dir":"alerts"})
    zones   =ZoneManager("config/zones.yaml","default")
    viz     =Visualizer()

    alert_cd    = GEN.get("alert_cooldown_seconds",cfg.get("fall",{}).get("cooldown_seconds",120))
    snapshot_dir = str(ROOT / "logs" / "snapshots")

    # ── Help Request Dispatcher เชื่อม MongoDB จริง ── NEW
    webhook_url = os.getenv("HELP_WEBHOOK_URL")
    if not webhook_url:
        print("⚠️  HELP_WEBHOOK_URL ไม่ได้ตั้งค่า — help request จะไม่ทำงาน")

    help_disp = HelpRequestDispatcher(
    webhook_url=webhook_url or "",  # บังคับเป็น str เสมอ
    )

    disp=AlertDispatcher(
        cooldowns={"fall":cfg.get("fall",{}).get("cooldown_seconds",alert_cd),
                   "hand_sos":cfg.get("hand_sos",{}).get("cooldown_seconds",alert_cd)},
        default_cooldown=alert_cd,
        enforce_one_per_file=GEN.get("one_alert_per_file",False),
        snapshot_dir=snapshot_dir,
        help_dispatcher=help_disp)

    hand_states={}
    hand_ev={}; hand_bc={}; hand_bf={}; hand_bt={}
    fall_ev={}; fall_bc={}; fall_bf={}; fall_bt={}
    s_states=defaultdict(lambda:{"angle_hist":deque(maxlen=TRIG_FRAMES),
                                  "motion_hist":deque(maxlen=TRIG_FRAMES),
                                  "triggered":False,"trigger_time":None,"frames_in_trigger":0})
    n=0
    try:
        while True:
            ret,raw=cap.read()
            if not ret:
                # ── ถ้า VLC ตาย ให้ restart แล้ว reconnect ─────────────────
                if vlc_mgr and not vlc_mgr.is_alive():
                    print(f"⚠️  [VLC] Process died — restarting ...")
                    try:
                        url = vlc_mgr.restart()
                        cap.release()
                        time.sleep(3)
                        cap = cv2.VideoCapture(url)
                    except Exception as restart_err:
                        print(f"❌ [VLC] Restart failed: {restart_err}")
                        break
                time.sleep(0.05); continue
            n+=1
            if n%max(1,SKIP)!=0: continue

            frame=preprocess(raw,W,H)
            h,w=frame.shape[:2]
            zones.draw(frame)

            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            try: hand_d.process_frame(rgb)
            except Exception: pass

            now_time=time.strftime("%Y-%m-%d %H:%M:%S")
            resp=_api("/detect_all",frame)
            pdets=resp.get("people",[])
            odets=resp.get("objects",[])

            # ── Object Guardian ──────────────────────────────────────────
            import uuid
            gp=[{"bbox":d["bbox"],"track_id":i} for i,d in enumerate(pdets)]
            for oa in obj_grd.update(frame,odets,gp,source_id=source_id,location=location):
                alert_logger.log_object_event(oa)
            obj_grd.draw(frame)

            # ── People tracking ──────────────────────────────────────────
            if pdets:
                assigned=tracker.update(pdets)
                boxes=BoxesWrapper([d["bbox"] for d in assigned],
                                   [d.get("conf",0.0) for d in assigned],
                                   [d.get("track_id") for d in assigned])
                kpts=KeypointsWrapper([d.get("keypoints",[]) for d in assigned])
            else:
                boxes=None; kpts=None

            if boxes and kpts and kpts.data:
                for i in range(len(boxes.xyxy)):
                    bbox=boxes.xyxy[i].cpu().numpy().tolist()
                    conf=float(boxes.conf[i].cpu())
                    tid=int(boxes.id[i].cpu()) if boxes.id else i
                    if i>=len(kpts.data): continue
                    kp=kpts.data[i].cpu().numpy()
                    x1,y1,x2,y2=bbox
                    if not zones.in_zone((x1+x2)/2/w,(y1+y2)/2/h): continue

                    fr=fall_d.process(tid,kp,bbox,h,w)


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
                            st.update(triggered=True,trigger_time=time.time(),frames_in_trigger=0)
                        if st["triggered"]:
                            st["frames_in_trigger"]+=1
                            if fr.get("recovered_quickly") or abs(avn)>mth*4:
                                st.update(triggered=False,trigger_time=None,frames_in_trigger=0)
                            elif st["trigger_time"] and (time.time()-st["trigger_time"])>CONF_WIN:
                                st.update(triggered=False,trigger_time=None,frames_in_trigger=0)
                    except Exception: pass

                    # ── Hand SOS ─────────────────────────────────────────
                    hs=hand_states.get(tid,0); hdet=False
                    try:
                        rh=getattr(hand_d,"_results",None)
                        if rh and getattr(rh,"hand_landmarks",None):
                            for hl in rh.hand_landmarks:
                                xs=[l.x for l in hl];ys=[l.y for l in hl]
                                area=(max(xs)-min(xs))*w*(max(ys)-min(ys))*h/(w*h)
                                if area<cfg.get("hand_sos",{}).get("min_hand_bbox_area_norm",0.002): continue
                                pts=[(int(l.x*w),int(l.y*h)) for l in hl]
                                hx=sum(p[0] for p in pts)/len(pts)
                                hy=sum(p[1] for p in pts)/len(pts)
                                if x1<=hx<=x2 and y1<=hy<=y2:
                                    hdet=True
                                    try:
                                        if hs==0 and hand_d._palm_open(hl): hs=1
                                        elif hs==1 and hand_d._thumb_in(hl): hs=2
                                        elif hs==2 and hand_d._fingers_closed(hl): hs=3
                                    except Exception: pass
                                    break
                    except Exception: pass
                    if not hdet: hs=max(0,hs-1)
                    hand_states[tid]=hs

                    if hs==3 and not fall_ev.get(tid):
                        if not hand_ev.get(tid) or conf>hand_bc.get(tid,0):
                            hand_bc[tid]=conf;hand_bf[tid]=raw.copy();hand_bt[tid]=now_time
                        hand_ev[tid]=True
                    elif hand_ev.get(tid):
                        if hand_bf.get(tid) is not None:
                            hand_bf[tid] = add_sos_badge(hand_bf[tid], "hand_sos", location, hand_bt[tid])
                            ex={"track_id":tid,"source":source_id,"location":location}
                            img_path = disp.dispatch("hand_sos",hand_bf[tid],ex)
                            if img_path:
                                alert_logger.log_sos_event(
                                    event_type="hand_sos",severity=2,severity_name="HIGH",
                                    source_id=source_id,source_path=src,location=location,
                                    track_id=tid,image_path=str(img_path),meta_path=None,flags=[],extra=ex)
                                try:
                                    insert_incident(
                                        event_uuid    = str(uuid.uuid4()),
                                        event_type    = "hand_sos",
                                        severity      = 2,
                                        severity_name = "HIGH",
                                        source_id     = source_id,
                                        location      = location,
                                        track_id      = tid,
                                        image_path    = str(img_path),
                                        extra         = {**ex, "confidence": conf}
                                    )
                                except Exception as e:
                                    print(f"[DB] hand_sos insert error: {e}")
                        hand_ev[tid]=False;hand_bc[tid]=0;hand_bf[tid]=None;hand_bt[tid]=None

                    # ── Fall CSV ─────────────────────────────────────────
                    if not fr.get("recovered_quickly"):
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

                    # ── Critical Fall & Confirmed Fall (dispatch once) ────────
                    esc=fr.get("danger_lying") and not fr.get("recovered_quickly")
                    is_critical=fr.get("is_critical") and not fr.get("recovered_quickly")
                    is_confirmed=(fr.get("is_fallen") or esc) and not fr.get("recovered_quickly")

                    if (is_critical or is_confirmed):
                        if not fall_ev.get(tid):
                            fall_bc[tid]=conf;fall_bf[tid]=raw.copy();fall_bt[tid]=now_time
                            fall_bf[tid] = add_sos_badge(fall_bf[tid], "fall", location, fall_bt[tid])
                            ex={"track_id":tid,"source":source_id,"location":location,
                                "recovered_quickly":fr.get("recovered_quickly"),"fall_result":fr}
                            if is_critical: ex["critical"]=True
                            if esc: ex["auto_escalated_immobile"]=True
                            img_path = disp.dispatch("fall",fall_bf[tid],ex)
                            if img_path:
                                lv,ln,flags=disp._assess_alert_level("fall",ex)
                                alert_logger.log_sos_event(
                                    event_type="fall",severity=lv,severity_name=ln,
                                    source_id=source_id,source_path=src,location=location,
                                    track_id=tid,image_path=str(img_path),meta_path=None,flags=flags,extra=ex)
                                try:
                                    insert_incident(
                                        event_uuid    = str(uuid.uuid4()),
                                        event_type    = "fall",
                                        severity      = lv,
                                        severity_name = ln,
                                        source_id     = source_id,
                                        location      = location,
                                        track_id      = tid,
                                        image_path    = str(img_path),
                                        extra         = {**ex, "confidence": conf}
                                    )
                                except Exception as e:
                                    print(f"[DB] fall insert error: {e}")
                            fall_ev[tid]=True
                    else:
                        if fall_ev.get(tid):
                            fall_ev[tid]=False;fall_bc[tid]=0;fall_bf[tid]=None;fall_bt[tid]=None

            viz.fps(frame)

    except KeyboardInterrupt:
        print(f"\n[{source_id[-30:]}] Stopped.")
    finally:
        cap.release(); hand_d.release()
        if vlc_mgr: vlc_mgr.stop()
        print(f"[{source_id[-30:]}] Done.")

def run_source(s):
    try: main(s["path"],s.get("port",8081),s.get("location",""))
    except Exception as e: print(f"[run_source] {s.get('id')} error: {e}")

if __name__=="__main__":
    try: multiprocessing.set_start_method("spawn",force=True)
    except RuntimeError: pass
    all_sources=yaml.safe_load((ROOT/"config/sources.yaml").read_text())["sources"]
    sources=[s for s in all_sources if s.get("enabled",True)]
    print(f"\n{'='*60}")
    print(f"🎬 Multi-Source Detection System")
    print(f"{'='*60}")
    print(f"Enabled: {len(sources)}/{len(all_sources)} sources")
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
    except KeyboardInterrupt: print("\n⚠️  Stopping …")
    finally:
        for _,p in procs:
            if p.is_alive(): p.terminate()
        print("Main exiting.")