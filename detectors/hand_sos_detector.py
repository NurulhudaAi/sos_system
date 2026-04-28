import ssl, urllib.request, cv2
from pathlib import Path
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

class HandSOSDetector:
    """
    Silent SOS 3 ขั้น:
      0 → ไม่มีมือ / reset
      1 → เปิดฝ่ามือ
      2 → พับนิ้วหัวแม่มือเข้าใน
      3 → ปิดนิ้ว 4 นิ้วทับ (SOS สมบูรณ์)
    """
    def __init__(self, cfg):
        self._state   = 0
        self._results = None
        self._enabled = False
        try:
            model_path = self._download_hand_model()
            base_opts  = mp_python.BaseOptions(
                model_asset_path=model_path)
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
        return all(lm[t].y < lm[p].y for t,p in zip([8,12,16,20],[6,10,14,18]))

    def _thumb_in(self, lm):
        px = (lm[5].x + lm[17].x) / 2
        return abs(lm[4].x - px) < abs(lm[12].x - px) * 0.55

    def _fingers_closed(self, lm):
        return all(lm[t].y > lm[m].y for t,m in zip([8,12,16,20],[5,9,13,17]))

    def process_frame(self, frame_rgb) -> dict:
        if not self._enabled:
            return {"detected": False, "state": 0}
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb)
        self._results = self._detector.detect(mp_image)
        if self._results.hand_landmarks:
            for hand in self._results.hand_landmarks:
                if   self._state==0 and self._palm_open(hand):     self._state=1
                elif self._state==1 and self._thumb_in(hand):      self._state=2
                elif self._state==2 and self._fingers_closed(hand):self._state=3
        else:
            self._state = max(0, self._state-1)
        return {"detected": self._state==3, "state": self._state}

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
