#!/usr/bin/env python3
"""
test_zones.py — Unit test for ZoneManager polygon & rectangle math,
detector filtering, and hot-reloading.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import ZoneManager, get_zone_manager


def test_polygon_and_detector_filtering():
    print("\n--- Test 1: Polygon & Detector Filtering ---")
    zm = ZoneManager(cam_id="test_cam_01")

    # Define 2 zones:
    # Zone 1: Triangle in top-left for "fall" only
    # Zone 2: Rectangle in bottom-right for "hand_sos" & "pose_sos"
    test_zones = [
        {
            "name": "Stairs Triangle",
            "type": "polygon",
            "detectors": ["fall"],
            "points": [
                [0.1, 0.1],
                [0.4, 0.1],
                [0.1, 0.4]
            ]
        },
        {
            "name": "Desk Counter",
            "type": "rectangle",
            "detectors": ["hand_sos", "pose_sos"],
            "x1": 0.6,
            "y1": 0.6,
            "x2": 0.9,
            "y2": 0.9
        }
    ]

    zm.reload_zones(test_zones)
    assert len(zm.zones) == 2, f"Expected 2 zones, got {len(zm.zones)}"

    # Test point inside Zone 1 (stairs triangle: [0.15, 0.15])
    assert zm.in_zone(0.15, 0.15, detector_type="fall") == True, "Point should be inside fall zone"
    assert zm.in_zone(0.15, 0.15, detector_type="hand_sos") == False, "Hand SOS should NOT trigger in fall-only zone"
    assert zm.in_zone(0.15, 0.15, detector_type="object_guardian") == False, "Object Guardian should NOT trigger here"

    # Test point outside Zone 1 (e.g. [0.35, 0.35] is outside the triangle)
    assert zm.in_zone(0.35, 0.35, detector_type="fall") == False, "Point outside triangle should return False"

    # Test point inside Zone 2 (desk counter: [0.75, 0.75])
    assert zm.in_zone(0.75, 0.75, detector_type="hand_sos") == True, "Hand SOS should trigger in desk counter"
    assert zm.in_zone(0.75, 0.75, detector_type="pose_sos") == True, "Pose SOS should trigger in desk counter"
    assert zm.in_zone(0.75, 0.75, detector_type="fall") == False, "Fall should NOT trigger in desk counter"

    print("✅ Test 1 Passed: Polygon math and detector isolation working perfectly!")


def test_empty_zones_fallback():
    print("\n--- Test 2: Empty Zones Fallback ---")
    zm = ZoneManager(cam_id="test_cam_02")
    zm.reload_zones([])

    # When no zones configured, everything is allowed (full frame active)
    assert zm.in_zone(0.05, 0.05, detector_type="fall") == True
    assert zm.in_zone(0.95, 0.95, detector_type="hand_sos") == True
    print("✅ Test 2 Passed: Empty zones allow full-frame monitoring.")


def test_hot_reload_registry():
    print("\n--- Test 3: Global Registry & Hot-Reload ---")
    zm = ZoneManager(cam_id="CAM_REG_TEST")
    reg_zm = get_zone_manager("CAM_REG_TEST")
    assert reg_zm is zm, "Instance must be registered in ZONE_MANAGERS"

    # Swap zones dynamically
    reg_zm.reload_zones([
        {"name": "New Full Zone", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "detectors": ["all"]}
    ])
    assert len(zm.zones) == 1
    assert zm.in_zone(0.5, 0.5, detector_type="fall") == True
    print("✅ Test 3 Passed: In-memory hot-reload registry working perfectly.")


if __name__ == "__main__":
    test_polygon_and_detector_filtering()
    test_empty_zones_fallback()
    test_hot_reload_registry()
    print("\n🎉 ALL ZONE TESTS PASSED!\n")
