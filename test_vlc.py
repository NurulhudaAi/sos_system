#!/usr/bin/env python3
"""
test_vlc.py — ทดสอบ VLC startup
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vlc_stream import VLCStreamManager

VIDEO_PATH = "/Users/nurulhudaadamishaq/Downloads/fall1.mp4"

print("\n" + "="*60)
print("🔧 Testing VLC Stream Manager")
print("="*60)

try:
    print(f"Video: {VIDEO_PATH}")
    print(f"Port:  8081")
    print()

    vlc = VLCStreamManager(src=VIDEO_PATH, width=1280, height=720, fps=10, port=8081)

    print("⏳ Starting VLC...")
    url = vlc.start(wait=4.0)

    print(f"✅ VLC started successfully!")
    print(f"📡 Stream URL: {url}")
    print()

    # Test health
    print("⏳ Testing stream health...")
    if vlc.health_check():
        print("✅ Stream is responding!")
    else:
        print("❌ Stream not responding")

    # Clean up
    print("\n⏳ Stopping VLC...")
    vlc.stop()
    print("✅ VLC stopped")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ Test passed!")
print("="*60 + "\n")
