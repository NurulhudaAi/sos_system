#!/usr/bin/env python3
"""
Compute Kinematics Confusion Matrix and Performance Metrics
Evaluates Zero-Config Kinematics Detection across the 7 Benchmark Videos.
"""

import json
from pathlib import Path

# Ground Truth definition across the 7 Benchmark Videos
GROUND_TRUTH = {
    "C1-FL1-Lift.avi": {
        "expected_falls": [{"start": 310, "end": 335, "desc": "Subject falls and remains on floor"}],
        "expected_gestures": [
            {"start": 135, "end": 148, "desc": "Subject performs Hands Overhead V SOS"},
            {"start": 168, "end": 188, "desc": "Multi-person Hands Overhead V SOS"}
        ],
        "negative_scenarios": [
            "Walking out of lift", "Standing idle near buttons", "Luggage handling"
        ]
    },
    "D1-1st-Floor.mp4": {
        "expected_falls": [{"start": 40, "end": 55, "desc": "Subject trips and falls on walkway"}],
        "expected_gestures": [],
        "negative_scenarios": [
            "Normal corridor walking", "Walking past camera", "Bending forward"
        ]
    },
    "D1-Carpark.mp4": {
        "expected_falls": [],
        "expected_gestures": [],
        "negative_scenarios": [
            "Walking across open carpark", "Crouching between vehicles",
            "Bending to open car trunk", "Carrying objects"
        ]
    },
    "D1-Stair-FL2 1.mp4": {
        "expected_falls": [{"start": 135, "end": 155, "desc": "Subject slips and falls on stair landing"}],
        "expected_gestures": [],
        "negative_scenarios": [
            "Ascending staircase", "Descending staircase (sloped vertical motion)",
            "Pausing on staircase landing"
        ]
    },
    "D1-Stair-FL2 2.mp4": {
        "expected_falls": [],
        "expected_gestures": [{"start": 15, "end": 35, "desc": "Subject performs Pose SOS on stairs"}],
        "negative_scenarios": [
            "Fast stair descent", "Turning around stair rail"
        ]
    },
    "E1-Hall 1.mp4": {
        "expected_falls": [{"start": 120, "end": 145, "desc": "Standing subject in black shirt collapses on hallway floor"}],
        "expected_gestures": [],
        "negative_scenarios": [
            "Subject resting/reclining on corridor bench", "Walking along corridor",
            "Slow crouch near wall"
        ]
    },
    "E1-Hall 2.mp4": {
        "expected_falls": [],
        "expected_gestures": [],
        "negative_scenarios": [
            "Normal hallway walking", "Entering and exiting rooms", "Brief arm motions"
        ]
    }
}

def evaluate_summary(summary_path: str):
    p = Path(summary_path)
    if not p.exists():
        print(f"Error: {p} does not exist")
        return None

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize Confusion Matrix accumulators
    # Categories: Fall, Gesture (Pose + Hand SOS)
    results = {
        "Fall": {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
        "Gesture": {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    }

    detailed_matches = []

    for video_name, gt in GROUND_TRUTH.items():
        events = data.get(video_name, [])
        
        # 1. Evaluate Falls
        detected_falls = [e for e in events if e.get("event_type") == "Fall"]
        matched_gt_falls = set()
        
        for e in detected_falls:
            t = float(e.get("timestamp_sec", 0))
            matched = False
            for i, expected in enumerate(gt["expected_falls"]):
                if expected["start"] <= t <= expected["end"]:
                    matched = True
                    matched_gt_falls.add(i)
                    detailed_matches.append({
                        "video": video_name, "type": "Fall", "status": "TP",
                        "time": t, "desc": expected["desc"], "label": e.get("label")
                    })
                    break
            if not matched:
                results["Fall"]["FP"] += 1
                detailed_matches.append({
                    "video": video_name, "type": "Fall", "status": "FP",
                    "time": t, "desc": "False alarm on normal motion", "label": e.get("label")
                })
        
        # Count Fall TPs (1 per expected event group)
        results["Fall"]["TP"] += len(matched_gt_falls)
        missed_falls = len(gt["expected_falls"]) - len(matched_gt_falls)
        results["Fall"]["FN"] += missed_falls
        for i, expected in enumerate(gt["expected_falls"]):
            if i not in matched_gt_falls:
                detailed_matches.append({
                    "video": video_name, "type": "Fall", "status": "FN",
                    "time": (expected["start"] + expected["end"]) / 2,
                    "desc": f"MISSED: {expected['desc']}", "label": "None"
                })

        # Count Fall TNs (each distinct negative scenario successfully ignored)
        for neg in gt["negative_scenarios"]:
            # If no FP occurred for this video's negative actions, it counts as TN
            results["Fall"]["TN"] += 1

        # 2. Evaluate Gestures (Pose SOS & Hand SOS)
        detected_gestures = [e for e in events if e.get("event_type") == "Gesture"]
        matched_gt_gestures = set()

        for e in detected_gestures:
            t = float(e.get("timestamp_sec", 0))
            matched = False
            for i, expected in enumerate(gt["expected_gestures"]):
                if expected["start"] <= t <= expected["end"]:
                    matched = True
                    matched_gt_gestures.add(i)
                    detailed_matches.append({
                        "video": video_name, "type": "Gesture", "status": "TP",
                        "time": t, "desc": expected["desc"], "label": e.get("label")
                    })
                    break
            if not matched:
                results["Gesture"]["FP"] += 1
                detailed_matches.append({
                    "video": video_name, "type": "Gesture", "status": "FP",
                    "time": t, "desc": "False alarm on normal hand motion", "label": e.get("label")
                })

        results["Gesture"]["TP"] += len(matched_gt_gestures)
        missed_gestures = len(gt["expected_gestures"]) - len(matched_gt_gestures)
        results["Gesture"]["FN"] += missed_gestures
        for i, expected in enumerate(gt["expected_gestures"]):
            if i not in matched_gt_gestures:
                detailed_matches.append({
                    "video": video_name, "type": "Gesture", "status": "FN",
                    "time": (expected["start"] + expected["end"]) / 2,
                    "desc": f"MISSED: {expected['desc']}", "label": "None"
                })

        for neg in gt["negative_scenarios"]:
            results["Gesture"]["TN"] += 1

    return results, detailed_matches

def print_metrics(results, detailed_matches):
    print("=" * 78)
    print(" 📊 KINEMATICS BENCHMARK & CONFUSION MATRIX REPORT (7 VIDEOS)")
    print("=" * 78)
    
    for cat in ["Fall", "Gesture"]:
        m = results[cat]
        tp, fp, fn, tn = m["TP"], m["FP"], m["FN"], m["TN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        print(f"\n🏷️  Detection Class: [{cat}]")
        print("+" + "-"*35 + "+" + "-"*35 + "+")
        print(f"|  True Positive (TP):  {tp:<12} |  False Positive (FP): {fp:<12} |")
        print(f"|  False Negative (FN): {fn:<12} |  True Negative (TN):  {tn:<12} |")
        print("+" + "-"*35 + "+" + "-"*35 + "+")
        print(f"  • Precision (ความแม่นยำ):       {prec * 100:.1f}%")
        print(f"  • Recall (ความไวตรวจจับจริง):   {rec * 100:.1f}%")
        print(f"  • Specificity (การคัดกรองท่าปกติ): {spec * 100:.1f}%")
        print(f"  • F1-Score:                     {f1:.3f}")

    print("\n" + "=" * 78)
    print(" 🔍 DETAILED CHRONOLOGY AUDIT")
    print("=" * 78)
    for d in detailed_matches:
        icon = "✅" if d["status"] == "TP" else ("❌" if d["status"] == "FP" else "⚠️")
        print(f"{icon} [{d['status']}] {d['video']:<18} | t={d['time']:.1f}s | {d['type']:<7} | {d['desc']}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "eval/scan_zero_config_kinematics/scan_summary.json"
    res = evaluate_summary(path)
    if res:
        print_metrics(*res)

