#!/usr/bin/env python3
"""
stream_reader.py — LatestFrameReader

[QA-P0-1/2] Root cause of "stutter on real cameras":
main loop was synchronous (read → HTTP detect → MediaPipe → read ...).
Whenever processing was slower than the camera FPS, frames piled up in
FFmpeg's internal buffer (CAP_PROP_BUFFERSIZE is ignored by the FFmpeg
backend), so the pipeline processed ever-older frames, H.264 decoding
smeared on gaps, and the stream eventually timed out / reconnected.

Fix: a background thread reads continuously and keeps ONLY the newest
frame plus its capture timestamp. The processing loop always gets the
freshest frame; stale frames are dropped instead of queued. The thread
also owns reconnection with exponential backoff.
"""
import os
import time
import logging
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("stream_reader")


class LatestFrameReader:
    def __init__(self, url: str, name: str = "", stall_timeout: float = 5.0):
        self.url = url
        self.name = name or str(url)[-30:]
        self.stall_timeout = stall_timeout
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame: Optional[np.ndarray] = None
        self._ts: float = 0.0
        self._seq = 0
        self._last_read_seq = 0
        self._request_reconnect = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.dropped = 0          # frames overwritten before being consumed
        self.reconnects = 0

    # ─── capture helpers ────────────────────────────────────────────────
    def _open(self) -> Optional[cv2.VideoCapture]:
        url = self.url
        if isinstance(url, str) and url.startswith("rtsp://"):
            # TCP = no UDP packet-loss smearing; low-delay flags; 5s socket timeout
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|stimeout;5000000"
            )
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def start(self) -> bool:
        self._cap = self._open()
        if self._cap is None:
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"reader-{self.name}")
        self._thread.start()
        return True

    def reopen(self, url: Optional[str] = None) -> None:
        """Switch to a new URL (e.g. after VLC restart). Thread reconnects cleanly."""
        with self._lock:
            if url:
                self.url = url
            self._request_reconnect = True

    def _loop(self) -> None:
        backoff = 1.0
        fails = 0
        while self._running:
            if self._request_reconnect:
                with self._lock:
                    if self._cap is not None:
                        try:
                            self._cap.release()
                        except Exception:
                            pass
                        self._cap = None
                    self._request_reconnect = False
                fails = 0

            cap = self._cap
            if cap is None:
                time.sleep(backoff)
                new_cap = self._open()
                if new_cap is not None:
                    with self._lock:
                        self._cap = new_cap
                    self.reconnects += 1
                    backoff = 1.0
                    print(f"[reader:{self.name}] reconnected (#{self.reconnects})")
                else:
                    backoff = min(backoff * 2, 15.0)
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails >= 25:  # ~persistent failure → drop and reconnect
                    print(f"[reader:{self.name}] read failing — reconnecting")
                    with self._lock:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        self._cap = None
                    fails = 0
                else:
                    time.sleep(0.02)
                continue
            fails = 0
            with self._cond:
                if self._seq > self._last_read_seq:
                    self.dropped += 1
                self._frame = frame
                self._ts = time.time()  # capture time — use as frame timestamp
                self._seq += 1
                self._cond.notify_all()

    # ─── consumer API ───────────────────────────────────────────────────
    def read(self, timeout: Optional[float] = None) -> Tuple[bool, Optional[np.ndarray], float]:
        """Block until a frame newer than the last one returned is available.

        Returns (ok, frame, capture_timestamp)."""
        timeout = self.stall_timeout if timeout is None else timeout
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > self._last_read_seq or not self._running,
                                       timeout=timeout):
                return False, None, 0.0
            if self._frame is None:
                return False, None, 0.0
            self._last_read_seq = self._seq
            return True, self._frame, self._ts

    def release(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
