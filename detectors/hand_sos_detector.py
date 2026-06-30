import ssl, urllib.request, cv2
from pathlib import Path
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


class HandSOSDetector:
    """
    Silent SOS 3 ขั้น (ปรับแก้ B2):
      0 → ไม่มีมือ / reset
      1 → เปิดฝ่ามือ
      2 → พับนิ้วหัวแม่มือเข้าใน
      3 → ปิดนิ้ว 4 นิ้วทับ (SOS สมบูรณ์)

    [B2] _state เดิมเป็น single global int — ทำให้ 2 คนในเฟรมเดียวกัน
    state machine รบกวนกัน. ตอนนี้ _state ถูกลบออก; state จัดการ
    per-track ใน main.py ผ่าน hand_states[tid] dict แทน.
    HandSOSDetector ทำหน้าที่แค่: process_frame() + expose _results
    + helper methods (_palm_open, _thumb_in, _fingers_closed).
    """
    def __init__(self, cfg):
        self._results = None
        self._enabled = False
        self._thumb_ratio = cfg.get("thumb_in_ratio", 0.55)  # [FIX Issue 8] configurable
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
        px = (lm[5].x + lm[17].x) / 2
        return abs(lm[4].x - px) < abs(lm[12].x - px) * self._thumb_ratio  # [FIX Issue 8]

    def _fingers_closed(self, lm):
        return all(lm[t].y > lm[m].y for t,m in zip([8,12,16,20],[5,9,13,17]))

    def process_frame(self, frame_rgb) -> dict:
        """รัน MediaPipe; เก็บ _results ไว้ให้ main.py อ่าน per-track"""
        if not self._enabled:
            self._results = None
            return {"detected": False}
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._results = self._detector.detect(mp_image)
        # ไม่ update _state ที่นี่อีกต่อไป (B2) — state จัดการ per-track ใน main.py
        return {"detected": False}

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
