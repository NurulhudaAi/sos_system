#!/usr/bin/env python3
"""
test_main_single.py — ทดสอบ main() function เดียว
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Override multiprocessing before importing main
import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

print("\n🔧 Testing single main() call\n")

try:
    from main import main
    print("⏳ Calling main('/Users/nurulhudaadamishaq/Downloads/fall1.mp4', 8081, 'ห้องนอน')")
    print()
    main("/Users/nurulhudaadamishaq/Downloads/fall1.mp4", 8081, "ห้องนอน")
except KeyboardInterrupt:
    print("\n\n✅ Interrupted (normal)")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
