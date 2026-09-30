#!/usr/bin/env python3
"""
Unit test for ObjectGuardian refactor (Steps 1 & 2):
1. Owner binding via proximity / wrist keypoints
2. Unattended timer counting from owner_left_at
3. object_left alert when left unattended >= left_behind_seconds
4. Owner returns before timeout -> reset timer, no alert
5. object_theft alert when bag disappears and suspect was nearby
6. Clean retrieval (no alert) when owner picks up bag
7. Background object without owner -> no alert
"""
import sys
from pathlib import Path
import numpy as np

# Ensure project root in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detectors.object_guardian import ObjectGuardian


def run_tests():
    print("=== Testing ObjectGuardian Steps 1 & 2 Refactor ===")
    
    cfg = {
        "left_behind_seconds": 10.0,
        "theft_window_seconds": 5.0,
        "ownership_confirm_seconds": 2.0,
        "proximity_norm": 0.25,
        "min_confidence": 0.3,
        "track_iou_threshold": 0.15,
        "track_max_missed": 5,
        "missing_confirm_frames": 2,
    }
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # ── Test 1: Proximity via Wrist Keypoint ──
    print("\n[Test 1] Proximity via Wrist Keypoint...")
    guardian = ObjectGuardian(cfg)
    obj_bbox = [100, 200, 150, 250]  # bag (50x50)
    
    # Person standing next to bag, wrist near bag
    kpts = [[0, 0, 0]] * 17
    kpts[10] = [120, 220, 0.9]  # right wrist inside bag bbox
    person_with_wrist = {
        "track_id": 1,
        "bbox": [50, 50, 180, 400],  # person height = 350
        "keypoints": kpts,
    }
    is_near, reason, dist = guardian._is_person_near(obj_bbox, person_with_wrist)
    assert is_near is True, f"Expected near via wrist, got {is_near}"
    assert reason == "wrist", f"Expected reason 'wrist', got {reason}"
    print("  ✓ Proximity via wrist keypoint passed!")

    # ── Test 2: Proximity via Edge Distance (Normalized) ──
    print("\n[Test 2] Proximity via Edge Distance...")
    # Person standing close (edge distance 30px, height 300px -> 0.10 <= 0.25)
    person_close = {
        "track_id": 2,
        "bbox": [180, 100, 250, 400],  # left edge = 180, obj right edge = 150 -> dx = 30
        "keypoints": [],
    }
    is_near, reason, dist = guardian._is_person_near(obj_bbox, person_close)
    assert is_near is True, f"Expected near via proximity, got {is_near}"
    assert reason == "proximity"
    print(f"  ✓ Proximity via edge distance passed (dist/height={dist:.3f})!")

    # Person far away (edge distance 200px, height 300px -> 0.66 > 0.25)
    person_far = {
        "track_id": 3,
        "bbox": [350, 100, 420, 400],
        "keypoints": [],
    }
    is_near, reason, dist = guardian._is_person_near(obj_bbox, person_far)
    assert is_near is False, f"Expected far person to NOT be near, got {is_near}"
    print("  ✓ Far person rejected correctly!")

    # ── Test 3: Owner Binding & Unattended Left Behind Alert ──
    print("\n[Test 3] Owner Binding & object_left Alert...")
    guardian = ObjectGuardian(cfg)
    obj = [{"bbox": obj_bbox, "confidence": 0.8, "class_name": "backpack"}]
    
    # Frame 0 to 2s: Person 1 is near -> becomes owner
    t = 0.0
    while t <= 2.2:
        alerts = guardian.update(dummy_frame, obj, [person_close], t_sec=t)
        assert len(alerts) == 0, "No alerts should trigger while owner is present"
        t += 0.5
        
    track = list(guardian._tracked.values())[0]
    assert track["owner_tid"] == 2, f"Expected owner_tid == 2, got {track['owner_tid']}"
    assert track["state"] == "OWNED"
    print("  ✓ Owner binding confirmed after 2.0s!")

    # Owner walks away at t=3.0s
    t = 3.0
    # Between t=3.0 and t=12.9 (duration < 10.0s): No alert
    while t < 13.0:
        alerts = guardian.update(dummy_frame, obj, [person_far], t_sec=t)
        assert len(alerts) == 0, f"Alert fired too early at t={t}!"
        t += 1.0

    # At t=13.5s (duration = 13.5 - 3.0 = 10.5s >= 10.0s left_behind_seconds):
    alerts = guardian.update(dummy_frame, obj, [person_far], t_sec=13.5)
    assert len(alerts) == 1, f"Expected 1 object_left alert, got {len(alerts)}"
    assert alerts[0]["event_type"] == "object_left"
    assert alerts[0]["owner_track_id"] == 2
    assert round(alerts[0]["seconds_unattended"], 1) == 10.5
    print("  ✓ object_left alert emitted correctly at exact left_behind_seconds!")

    # ── Test 4: Owner Returns Before Timeout (Reset Timer) ──
    print("\n[Test 4] Owner Returns Before Timeout (Reset Timer)...")
    guardian = ObjectGuardian(cfg)
    t = 0.0
    while t <= 2.2:
        guardian.update(dummy_frame, obj, [person_close], t_sec=t)
        t += 0.5
    # Owner walks away for 5 seconds (not yet 10s)
    while t <= 7.0:
        guardian.update(dummy_frame, obj, [person_far], t_sec=t)
        t += 0.5
    # Owner returns!
    guardian.update(dummy_frame, obj, [person_close], t_sec=7.5)
    track = list(guardian._tracked.values())[0]
    assert track["owner_left_at"] is None, "owner_left_at should reset to None when owner returns"
    assert track["state"] == "OWNED"
    print("  ✓ Owner return reset unattended timer correctly!")

    # ── Test 5: Theft Detection (Suspect picks up bag) ──
    print("\n[Test 5] Theft Detection...")
    guardian = ObjectGuardian(cfg)
    # Establish owner
    t = 0.0
    while t <= 2.2:
        guardian.update(dummy_frame, obj, [person_close], t_sec=t)
        t += 0.5
    # Owner walks away
    t = 3.0
    guardian.update(dummy_frame, obj, [person_far], t_sec=t)
    
    # Suspect (Person 99) approaches bag
    suspect = {
        "track_id": 99,
        "bbox": [110, 150, 160, 380],  # near bag
        "keypoints": [],
    }
    # Suspect is at bag
    guardian.update(dummy_frame, obj, [suspect, person_far], t_sec=3.5)
    
    # Bag disappears (picked up by suspect)
    alerts = []
    # 2 frames missing
    guardian.update(dummy_frame, [], [suspect, person_far], t_sec=4.0)
    alerts = guardian.update(dummy_frame, [], [suspect, person_far], t_sec=4.5)
    
    assert len(alerts) == 1, f"Expected 1 object_theft alert, got {len(alerts)}"
    assert alerts[0]["event_type"] == "object_theft"
    assert alerts[0]["owner_track_id"] == 2
    assert alerts[0]["suspect_track_id"] == 99
    print("  ✓ object_theft alert correctly identified suspect 99 and owner 2!")

    # ── Test 6: Clean Retrieval (Owner picks up bag) ──
    print("\n[Test 6] Clean Retrieval (Owner picks up own bag)...")
    guardian = ObjectGuardian(cfg)
    t = 0.0
    while t <= 2.2:
        guardian.update(dummy_frame, obj, [person_close], t_sec=t)
        t += 0.5
    # Owner takes bag (bag disappears while owner is near)
    guardian.update(dummy_frame, [], [person_close], t_sec=2.5)
    alerts = guardian.update(dummy_frame, [], [person_close], t_sec=3.0)
    assert len(alerts) == 0, f"Expected 0 alerts for owner retrieval, got {len(alerts)}"
    print("  ✓ Owner retrieval remained completely silent (no false alarm)!")

    # ── Test 7: Background Object Without Owner ──
    print("\n[Test 7] Background Object Without Owner...")
    guardian = ObjectGuardian(cfg)
    # Bag sitting alone with no person nearby for 30 seconds
    t = 0.0
    while t <= 30.0:
        alerts = guardian.update(dummy_frame, obj, [], t_sec=t)
        assert len(alerts) == 0, f"Background object should not trigger alert, got {alerts}"
        t += 1.0
    print("  ✓ Background object never triggered false alert!")

    print("\n>>> ALL 7 UNIT TESTS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    run_tests()
