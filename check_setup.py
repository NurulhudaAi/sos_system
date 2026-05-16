import sys

print("=" * 50)
print("SOS System — Setup Check")
print("=" * 50)

# Python
print(f"\nPython: {sys.version.split()[0]}")

# OpenCV
try:
    import cv2
    print(f"OpenCV:     {cv2.__version__} ✓")
except ImportError as e:
    print(f"OpenCV:     FAIL — {e}")

# NumPy
try:
    import numpy as np
    print(f"NumPy:      {np.__version__} ✓")
except ImportError as e:
    print(f"NumPy:      FAIL — {e}")

# PyTorch + MPS
try:
    import torch
    mps = torch.backends.mps.is_available()
    print(f"PyTorch:    {torch.__version__} ✓")
    print(f"Apple MPS:  {'✓ (ใช้ M1 GPU ได้)' if mps else '✗ (ใช้ CPU)'}")
except ImportError as e:
    print(f"PyTorch:    FAIL — {e}")

# Ultralytics (YOLOv8)
try:
    from ultralytics import YOLO
    print(f"YOLOv8:     ✓")
except ImportError as e:
    print(f"YOLOv8:     FAIL — {e}")

# MediaPipe
try:
    import mediapipe as mp
    print(f"MediaPipe:  ✓")
except ImportError as e:
    print(f"MediaPipe:  FAIL — {e}")

# lap (Ultralytics tracker dependency)
try:
    import lap
    print(f"lap:        {lap.__version__} ✓")
except ImportError as e:
    print(f"lap:        FAIL — {e}")

# PyYAML
try:
    import yaml
    print(f"PyYAML:     ✓")
except ImportError as e:
    print(f"PyYAML:     FAIL — {e}")

# VLC command
import subprocess
import shutil
import platform

if platform.system() == "Darwin":
    vlc_cmd = "/Applications/VLC.app/Contents/MacOS/VLC"
else:
    vlc_cmd = shutil.which("vlc")

if vlc_cmd:
    try:
        r = subprocess.run([vlc_cmd, "--version"],
                           capture_output=True, text=True, timeout=5)
        ver = r.stdout.split("\n")[0]
        print(f"VLC:        {ver} ✓")
    except Exception as e:
        print(f"VLC:        FAIL — {e}")
else:
    print("VLC:        FAIL — VLC not found in PATH")

print("\n" + "=" * 50)
print("ถ้าทุกอย่างขึ้น ✓ พร้อมไปขั้นถัดไป")
print("=" * 50)
