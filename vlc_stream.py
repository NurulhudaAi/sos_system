#!/usr/bin/env python3
"""
vlc_stream.py
VLCStreamManager — แปลง RTSP / ไฟล์วิดีโอ เป็น HTTP stream
ให้ OpenCV อ่านได้ผ่าน localhost
"""
import subprocess
import time
import logging
import socket
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _free_port(default: int) -> int:
    """หา port ที่ว่างอยู่"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", default))
            return default
        except OSError:
            s.bind(("", 0))
            return s.getsockname()[1]


class VLCStreamManager:
    """
    เปิด VLC ใน background เพื่อ transcode
    - RTSP camera  → HTTP MJPEG stream
    - ไฟล์วิดีโอ   → HTTP MJPEG stream
    แล้วคืน URL ให้ OpenCV เปิดต่อ
    """

    def __init__(
        self,
        src:    str,
        width:  int  = 1280,
        height: int  = 720,
        fps:    int  = 10,
        port:   int  = 8081,
    ):
        self.src    = src
        self.width  = width
        self.height = height
        self.fps    = fps
        self.port   = _free_port(port)
        self._proc: Optional[subprocess.Popen] = None

    # ─── Build VLC command ────────────────────────────────────────────────────

    def _build_cmd(self) -> list:
        sout = (
            f"#transcode{{vcodec=MJPG,vb=800,scale=1,"
            f"width={self.width},height={self.height},fps={self.fps},"
            f"acodec=none}}"
            f":std{{access=http,mux=mpjpeg,dst=0.0.0.0:{self.port}/}}"
        )

        src_path = Path(self.src)
        input_src = str(src_path.resolve()) if src_path.exists() else self.src

        vlc_bin = "/Applications/VLC.app/Contents/MacOS/VLC"
        if not Path(vlc_bin).exists():
            vlc_bin = "vlc"

        cmd = [
            vlc_bin, "-I", "dummy",
            "--no-audio",
            "--loop",
            input_src,
            f"--sout={sout}",
            "--sout-keep",
        ]
        return cmd

    # ─── Start / Stop ─────────────────────────────────────────────────────────

    def start(self, wait: float = 3.0) -> str:
        """เริ่ม VLC แล้วคืน URL สำหรับ OpenCV"""
        cmd = self._build_cmd()
        logger.info(f"[VLC] Starting: {' '.join(cmd)}")

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "VLC ไม่ได้ติดตั้ง — ติดตั้งด้วย: brew install vlc (Mac) หรือ apt install vlc (Linux)"
            )

        # รอให้ VLC พร้อม
        time.sleep(wait)

        if self._proc.poll() is not None:
            stderr_bytes = b""
            try:
                stderr_bytes = self._proc.stderr.read() if self._proc.stderr else b""
            except Exception:
                pass
            error_msg = stderr_bytes.decode("utf-8", errors="ignore") or "unknown error"
            raise RuntimeError(
                f"VLC หยุดทำงานก่อนกำหนด (exit code {self._proc.returncode})\n"
                f"Error: {error_msg[:500]}"
            )

        url = f"http://localhost:{self.port}/"
        logger.info(f"[VLC] Stream ready: {url}")
        return url

    def stop(self):
        """หยุด VLC"""
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            logger.info("[VLC] Stopped")
        self._proc = None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def restart(self) -> str:
        """Restart VLC แล้วคืน URL ใหม่"""
        logger.warning("[VLC] Restarting...")
        self.stop()
        return self.start()

    # ─── Health check ─────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """เช็คว่า stream ยังตอบสนองอยู่ไหม"""
        import urllib.request
        try:
            urllib.request.urlopen(
                f"http://localhost:{self.port}/",
                timeout=3
            )
            return True
        except Exception:
            return False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()