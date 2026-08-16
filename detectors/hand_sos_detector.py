import ssl, urllib.request, cv2
from pathlib import Path
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


class HandSOSDetector:
    """
    Silent SOS 3 ขั้น (ปรับแก้ B2 + RTSP fix):
      0 → ไม่มีมือ / reset
      1 → เปิดฝ่ามือ
      2 → พับนิ้วหัวแม่มือเข้าใน
      3 → ปิดนิ้ว 4 นิ้วทับ (SOS สมบูรณ์)

    [B2] _state เดิมเป็น single global int — ทำให้ 2 คนในเฟรมเดียวกัน
    state machine รบกวนกัน. ตอนนี้ _state ถูกลบออก; state จัดการ
    per-track ใน main.py ผ่าน hand_states[tid] dict แทน.
    HandSOSDetector ทำหน้าที่แค่: process_frame() + expose _results
    + helper methods (_palm_open, _thumb_in, _fingers_closed)
    + check_sos_step() (state machine helper for callers).

    [RTSP FIX] ปรับปรุงสำหรับภาพจากกล้อง RTSP ระยะไกล:
    - _fingers_closed() มี 3-tier fallback (distance → y-based → curvature)
    - _thumb_and_fingers_closed() combined check สำหรับท่ากำมือ
    - check_sos_step() รวม state machine logic + allow_fast_sos
    - process_crop() สำหรับ per-person crop strategy
    """
    def __init__(self, cfg):
        self._results = None
        self._enabled = False
        self._thumb_ratio = cfg.get("thumb_in_ratio", 0.55)  # [FIX Issue 8] configurable
        self._fingers_closed_ratio = cfg.get("fingers_closed_ratio", 1.40)  # [RTSP FIX] was 1.25
        self._fingers_closed_min = cfg.get("fingers_closed_min_count", 2)   # [RTSP FIX] was 3
        self._allow_fast_sos = cfg.get("allow_fast_sos", True)  # [RTSP FIX] state 1→3 combined
        # [FIX-FP] strict ratio สำหรับ fast_sos path เท่านั้น (เข้มกว่า ratio ทั่วไป)
        self._fingers_closed_strict_ratio = cfg.get("fingers_closed_strict_ratio", 1.20)
        # [FIX-FP] minimum hand span ก่อนอนุญาต fast_sos (กัน noise keypoint)
        self._fast_sos_min_hand_span = cfg.get("fast_sos_min_hand_span", 0.04)
        try:
            model_path = self._download_hand_model()
            base_opts  = mp_python.BaseOptions(model_asset_path=model_path)
            opts = mp_vision.HandLandmarkerOptions(
                base_options=base_opts,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=cfg.get("min_detection_confidence", 0.70),
                min_hand_presence_confidence=cfg.get("min_tracking_confidence",  0.50),
                min_tracking_confidence=cfg.get("min_tracking_confidence",  0.50),
            )
            self._detector = mp_vision.HandLandmarker.create_from_options(opts)
            self._enabled  = True
            print("HandSOSDetector ready ✓")
        except Exception as e:
            print(f"[WARN] HandSOSDetector disabled: {e}")

    def _download_hand_model(self):
        path = Path("models/hand_landmarker.task")
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            print("Downloading hand landmark model (~8MB)...")
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "hand_landmarker/hand_landmarker/float16/1/"
                   "hand_landmarker.task")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(url, context=ctx) as r, open(str(path),"wb") as f:
                f.write(r.read())
            print("Model downloaded ✓")
        return str(path)

    def _palm_open(self, lm):
        # [FIX Issue 9] Primary: fingertip above PIP joint (hand facing camera)
        y_check = all(lm[t].y < lm[p].y for t,p in zip([8,12,16,20],[6,10,14,18]))
        if y_check:
            return True
        # Fallback: distance-based check for side-facing hands
        # fingertip-to-MCP distance > PIP-to-MCP distance × 1.3 means finger is extended
        extended = sum(
            1 for t, m, p in zip([8,12,16,20], [5,9,13,17], [6,10,14,18])
            if ((lm[t].x-lm[m].x)**2 + (lm[t].y-lm[m].y)**2)**0.5 >
               ((lm[p].x-lm[m].x)**2 + (lm[p].y-lm[m].y)**2)**0.5 * 1.3
        )
        return extended >= 3  # อย่างน้อย 3 ใน 4 นิ้วยืดออก

    def _thumb_in(self, lm):
        """ระยะห่างจาก thumb tip ถึง palm center เทียบกับ hand span — ทนการหมุนมือ
        [FIX] เพิ่ม fallback: ถ้า thumb tip ใกล้ index MCP (lm[5]) มาก ก็ถือว่า tucked
        """
        px = (lm[5].x + lm[17].x) / 2
        py = (lm[5].y + lm[17].y) / 2
        hand_span = ((lm[5].x - lm[17].x)**2 + (lm[5].y - lm[17].y)**2) ** 0.5
        thumb_dist = ((lm[4].x - px)**2 + (lm[4].y - py)**2) ** 0.5
        # Primary check: thumb close to palm center
        if thumb_dist < hand_span * self._thumb_ratio:
            return True
        # Fallback: thumb tip close to index finger MCP (tucked under fingers)
        thumb_to_index = ((lm[4].x - lm[5].x)**2 + (lm[4].y - lm[5].y)**2) ** 0.5
        return thumb_to_index < hand_span * 0.45

    def _fingers_closed(self, lm):
        """[RTSP FIX] 3-tier check สำหรับนิ้วปิด — ทนภาพ RTSP ระยะไกล

        Tier 1 (distance-based): fingertip→wrist < MCP→wrist × ratio
        Tier 2 (y-based fallback): fingertip.y > PIP.y (นิ้วงอลง)
        Tier 3 (curl-based fallback): fingertip ใกล้ MCP มากกว่า PIP (นิ้วม้วนเข้า)

        ต้องผ่าน ≥ min_count (default 2) ใน 4 นิ้ว
        """
        wrist = lm[0]
        def _d(a, b):
            return ((a.x-b.x)**2 + (a.y-b.y)**2) ** 0.5

        tips = [8, 12, 16, 20]
        mcps = [5, 9, 13, 17]
        pips = [6, 10, 14, 18]

        closed_count = 0
        for t, m, p in zip(tips, mcps, pips):
            # Tier 1: distance-based (original)
            if _d(lm[t], wrist) < _d(lm[m], wrist) * self._fingers_closed_ratio:
                closed_count += 1
                continue
            # Tier 2: y-based — fingertip below PIP means finger curled (camera facing)
            if lm[t].y > lm[p].y and lm[t].y > lm[m].y:
                closed_count += 1
                continue
            # Tier 3: curl — fingertip closer to MCP than PIP is to MCP
            # (finger tip has curled past the PIP joint)
            tip_to_mcp = _d(lm[t], lm[m])
            pip_to_mcp = _d(lm[p], lm[m])
            if pip_to_mcp > 1e-6 and tip_to_mcp < pip_to_mcp * 1.1:
                closed_count += 1
                continue

        return closed_count >= self._fingers_closed_min

    def _fingers_closed_strict(self, lm):
        """[FIX-FP] Strict version: Tier 1 only, ratio 1.20, ≥4 fingers.
        ใช้สำหรับ fast_sos path (state 1→3) เท่านั้น — เข้มกว่า _fingers_closed()
        """
        wrist = lm[0]
        def _d(a, b):
            return ((a.x-b.x)**2 + (a.y-b.y)**2) ** 0.5
        tips = [8, 12, 16, 20]
        mcps = [5, 9, 13, 17]
        closed_count = sum(
            1 for t, m in zip(tips, mcps)
            if _d(lm[t], wrist) < _d(lm[m], wrist) * self._fingers_closed_strict_ratio
        )
        return closed_count >= 4  # [FIX-FP] ต้องครบ 4/4 นิ้ว

    def _thumb_and_fingers_closed(self, lm):
        """[FIX-FP] Combined check: thumb tucked + fingers closed (strict, Tier 1 only)
        + not palm_open — ลดโอกาส fingers_closed กับ palm_open True พร้อมกัน"""
        return (
            self._thumb_in(lm)
            and not self._palm_open(lm)
            and self._fingers_closed_strict(lm)
        )

    def check_sos_step(self, cur_state, lm):
        """[RTSP FIX] State machine helper — ใช้แทนการเขียน state transition
        ซ้ำใน main.py/live_full_rtsp.py

        Args:
            cur_state: สถานะปัจจุบัน (0-3)
            lm: hand landmarks (21 จุด) หรือ None ถ้าไม่เจอมือ

        Returns:
            new_state: สถานะใหม่ (0-3)
        """
        if lm is None:
            return cur_state  # ไม่เปลี่ยน state ถ้าไม่เจอมือ (grace จัดการที่ caller)

        if cur_state == 0:
            if self._palm_open(lm):
                return 1
            return 0

        if cur_state == 1:
            # [FIX-FP] fast_sos: strict ratio 1.20 + 4/4 นิ้ว + hand_span gate + not palm_open
            if self._allow_fast_sos:
                hand_span = ((lm[5].x - lm[17].x)**2 + (lm[5].y - lm[17].y)**2) ** 0.5
                if (hand_span >= self._fast_sos_min_hand_span
                        and self._fingers_closed_strict(lm)
                        and not self._palm_open(lm)):
                    return 3
            if self._thumb_and_fingers_closed(lm):
                return 3
            if self._thumb_in(lm):
                return 2
            return 1

        if cur_state == 2:
            # [FIX-FP] เพิ่ม not _palm_open() — กัน Tier 2/3 ผ่านขณะมือยังเปิด
            if self._fingers_closed(lm) and not self._palm_open(lm):
                return 3
            # [RTSP FIX] ถ้า thumb ยังอยู่ใน + เจอมือ → ค้างที่ 2 (ไม่ decay)
            return 2

        return cur_state  # state 3 → ค้างไว้ให้ caller reset

    def get_debug_info(self, lm):
        """[RTSP FIX] คืนค่า debug สำหรับ skeleton debugging tool

        Returns dict ของค่าทุกตัวที่ใช้ตัดสินใจ — สำหรับ log/overlay
        """
        if lm is None:
            return {}

        wrist = lm[0]
        def _d(a, b):
            return ((a.x-b.x)**2 + (a.y-b.y)**2) ** 0.5

        px = (lm[5].x + lm[17].x) / 2
        py = (lm[5].y + lm[17].y) / 2
        hand_span = _d(lm[5], lm[17])
        thumb_dist = _d(lm[4], type('P', (), {'x': px, 'y': py})())
        thumb_to_idx = _d(lm[4], lm[5])

        tips = [8, 12, 16, 20]
        mcps = [5, 9, 13, 17]
        pips = [6, 10, 14, 18]
        finger_names = ['index', 'middle', 'ring', 'pinky']

        finger_details = {}
        for name, t, m, p in zip(finger_names, tips, mcps, pips):
            tip_wrist = _d(lm[t], wrist)
            mcp_wrist = _d(lm[m], wrist)
            tip_mcp = _d(lm[t], lm[m])
            pip_mcp = _d(lm[p], lm[m])
            finger_details[name] = {
                'tip_wrist': round(tip_wrist, 4),
                'mcp_wrist': round(mcp_wrist, 4),
                'ratio': round(tip_wrist / max(mcp_wrist, 1e-6), 3),
                'tip_y': round(lm[t].y, 4),
                'pip_y': round(lm[p].y, 4),
                'mcp_y': round(lm[m].y, 4),
                'tip_mcp': round(tip_mcp, 4),
                'pip_mcp': round(pip_mcp, 4),
                't1_closed': tip_wrist < mcp_wrist * self._fingers_closed_ratio,
                't2_closed': lm[t].y > lm[p].y and lm[t].y > lm[m].y,
                't3_closed': pip_mcp > 1e-6 and tip_mcp < pip_mcp * 1.1,
            }

        return {
            'palm_open': self._palm_open(lm),
            'thumb_in': self._thumb_in(lm),
            'fingers_closed': self._fingers_closed(lm),
            'combined': self._thumb_and_fingers_closed(lm),
            'hand_span': round(hand_span, 4),
            'thumb_dist': round(thumb_dist, 4),
            'thumb_ratio': round(thumb_dist / max(hand_span, 1e-6), 3),
            'thumb_to_idx': round(thumb_to_idx, 4),
            'thumb_idx_ratio': round(thumb_to_idx / max(hand_span, 1e-6), 3),
            'fingers': finger_details,
        }

    def process_frame(self, frame_rgb) -> dict:
        """รัน MediaPipe; เก็บ _results ไว้ให้ main.py อ่าน per-track"""
        if not self._enabled:
            self._results = None
            return {"detected": False}
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._results = self._detector.detect(mp_image)
        # ไม่ update _state ที่นี่อีกต่อไป (B2) — state จัดการ per-track ใน main.py
        return {"detected": False}

    def process_crop(self, frame_bgr, bbox, min_crop_h=256):
        """[RTSP FIX] Crop person bbox → resize → detect hands — สำหรับภาพ RTSP

        Args:
            frame_bgr: full frame (BGR)
            bbox: [x1, y1, x2, y2] person bounding box
            min_crop_h: minimum crop height for MediaPipe (default 256px)

        Returns:
            list of hand landmarks (normalized to crop coords) or empty list
        """
        if not self._enabled:
            self._results = None
            return []

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Expand bbox by 10% for hand margin
        bw, bh = x2 - x1, y2 - y1
        cx1 = max(0, x1 - int(bw * 0.1))
        cy1 = max(0, y1 - int(bh * 0.1))
        cx2 = min(w, x2 + int(bw * 0.1))
        cy2 = min(h, y2 + int(bh * 0.1))

        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            self._results = None
            return []

        ch, cw = crop.shape[:2]
        # Resize crop to at least min_crop_h for MediaPipe
        if ch < min_crop_h and ch > 0:
            scale = min_crop_h / ch
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)))

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
        self._results = self._detector.detect(mp_image)

        if self._results and self._results.hand_landmarks:
            return list(self._results.hand_landmarks)
        return []

    def draw(self, frame, result):
        if not self._enabled or not self._results:
            return
        if not self._results.hand_landmarks:
            return
        h, w = frame.shape[:2]
        for hand in self._results.hand_landmarks:
            pts = [(int(lm.x*w), int(lm.y*h)) for lm in hand]
            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
                (5,9),(9,13),(13,17),
            ]
            for a,b in connections:
                cv2.line(frame, pts[a], pts[b], (0,200,0), 1)
            for pt in pts:
                cv2.circle(frame, pt, 3, (0,255,0), -1)

    def release(self):
        if self._enabled:
            self._detector.close()
