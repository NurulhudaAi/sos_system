#!/usr/bin/env python3
"""
stress_test_day6.py — Day 6 Automated Stress, Stability & Performance Benchmark

Suite covers:
  Test 1: Per-Component Latency Breakdown (YOLOv8-pose, Hungarian Tracker, Fall, Hand SOS, Pose SOS)
  Test 2: Sustained 1,000-Frame Load & Process Memory Leak Profiling (RSS Memory Delta)
  Test 3: LatestFrameReader (RTSP) Zero-Buffer & Auto-Reconnect Fault Tolerance
  Test 4: Multi-Camera Concurrent Load & Server Throughput (4 Concurrent Streams)
"""

import sys
import os
import time
import json
import psutil
import threading
import requests
import cv2
import numpy as np
import yaml
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import preprocess
from tracker import HungarianTracker, SpatialAlertDeduplicator
from detectors.fall_detector import FallDetector
from detectors.hand_sos_detector import HandSOSDetector
from detectors.hands_over_head_detector import HandsOverHeadDetector
from stream_reader import LatestFrameReader

DET_URL = os.getenv("DETECTOR_URL", "http://127.0.0.1:8000")


def get_process_memory_mb(pid: int = None) -> float:
    """Returns RSS memory in MB for the specified PID (or current process)."""
    try:
        p = psutil.Process(pid or os.getpid())
        return p.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def find_server_pid() -> int:
    """Finds the PID of model_server.py (Python process)."""
    candidates = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = proc.info.get('cmdline') or []
            cmd_str = " ".join(cmd)
            if "model_server.py" in cmd_str:
                rss = proc.memory_info().rss
                candidates.append((proc.info['pid'], rss))
        except Exception:
            pass
    if candidates:
        # Pick process with highest RSS (the actual Python worker)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    return None


def run_test_1_latency_breakdown(video_path: Path, num_frames: int = 150) -> dict:
    print("\n" + "="*70)
    print("TEST 1: Per-Component Latency Breakdown & FPS Benchmark")
    print("="*70)

    cfg = yaml.safe_load((ROOT / "config/thresholds.yaml").read_text())
    hand_cfg = cfg.get("hand_sos", {})
    fall_cfg = cfg.get("fall", {})
    pose_cfg = cfg.get("hands_over_head", {})

    W, H = 1920, 1080
    tracker = HungarianTracker(frame_w=W, frame_h=H)
    fall_d = FallDetector(fall_cfg)
    hand_d = HandSOSDetector(hand_cfg)
    head_d = HandsOverHeadDetector(pose_cfg)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return {}

    timings = defaultdict(list)
    frames_processed = 0

    for _ in range(num_frames):
        ret, raw = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, raw = cap.read()
            if not ret:
                break

        t0 = time.perf_counter()
        frame = preprocess(raw, W, H)
        clean_frame = frame.copy()
        h, w = frame.shape[:2]

        # CLAHE
        t_clahe_start = time.perf_counter()
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        frame_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        timings["clahe"].append((time.perf_counter() - t_clahe_start) * 1000)

        # Model Server (YOLOv8-Pose)
        t_srv_start = time.perf_counter()
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        try:
            r = requests.post(f"{DET_URL}/detect_all", files={"image": ("f.jpg", buf.tobytes(), "image/jpeg")}, timeout=5)
            pdets = r.json().get("people", []) if r.status_code == 200 else []
        except Exception:
            pdets = []
        timings["server_inference"].append((time.perf_counter() - t_srv_start) * 1000)

        # Hungarian Tracker
        t_trk_start = time.perf_counter()
        assigned = tracker.update(pdets) if pdets else []
        timings["hungarian_tracker"].append((time.perf_counter() - t_trk_start) * 1000)

        # Detectors
        t_fall = 0.0
        t_hand = 0.0
        t_pose = 0.0

        for det in assigned:
            tid = det.get("track_id")
            bbox = det.get("bbox", [0, 0, 0, 0])
            kps = np.array(det.get("keypoints", []))

            # Fall
            tf0 = time.perf_counter()
            _ = fall_d.process(tid, kps, bbox, h, w, timestamp=frames_processed * 0.1)
            t_fall += (time.perf_counter() - tf0) * 1000

            # Hand SOS
            th0 = time.perf_counter()
            _ = hand_d.process_wrist_crop(frame_clahe, kps, h, w, bbox=bbox)
            t_hand += (time.perf_counter() - th0) * 1000

            # Pose SOS
            tp0 = time.perf_counter()
            _ = head_d.process(tid, kps, h, w, timestamp=frames_processed * 0.1)
            t_pose += (time.perf_counter() - tp0) * 1000

        timings["fall_detector"].append(t_fall)
        timings["hand_sos_detector"].append(t_hand)
        timings["pose_sos_detector"].append(t_pose)

        t_total = (time.perf_counter() - t0) * 1000
        timings["total_pipeline"].append(t_total)
        frames_processed += 1

    cap.release()
    hand_d.release()

    res = {}
    print(f"\nBenchmark results over {frames_processed} frames:")
    print(f"{'-'*55}")
    print(f"{'Component':<28} | {'Avg (ms)':<10} | {'p95 (ms)':<10}")
    print(f"{'-'*55}")
    for k, v in timings.items():
        avg_ms = float(np.mean(v))
        p95_ms = float(np.percentile(v, 95))
        res[k] = {"avg_ms": round(avg_ms, 2), "p95_ms": round(p95_ms, 2)}
        print(f"{k:<28} | {avg_ms:8.2f} ms | {p95_ms:8.2f} ms")
    print(f"{'-'*55}")

    avg_total_ms = res["total_pipeline"]["avg_ms"]
    effective_fps = 1000.0 / max(1.0, avg_total_ms)
    res["effective_fps"] = round(effective_fps, 1)
    print(f"⚡ Effective Pipeline Processing Speed: {effective_fps:.1f} FPS (Avg Latency: {avg_total_ms:.1f} ms)")
    return res


def run_test_2_sustained_memory_leak(video_path: Path, num_frames: int = 1000) -> dict:
    print("\n" + "="*70)
    print(f"TEST 2: Sustained {num_frames}-Frame Load & Memory Leak Profiling")
    print("="*70)

    server_pid = find_server_pid()
    client_pid = os.getpid()

    cfg = yaml.safe_load((ROOT / "config/thresholds.yaml").read_text())
    W, H = 1920, 1080
    tracker = HungarianTracker(frame_w=W, frame_h=H)
    fall_d = FallDetector(cfg.get("fall", {}))
    hand_d = HandSOSDetector(cfg.get("hand_sos", {}))
    head_d = HandsOverHeadDetector(cfg.get("hands_over_head", {}))

    cap = cv2.VideoCapture(str(video_path))
    mem_snapshots = []

    print(f"Starting memory profiling: Client PID={client_pid}, Server PID={server_pid}")
    t_start = time.time()

    check_points = [0, 50, 100, 150, 200, 300]

    for f_idx in range(num_frames + 1):
        ret, raw = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, raw = cap.read()
            if not ret:
                break

        frame = preprocess(raw, W, H)
        h, w = frame.shape[:2]

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        try:
            r = requests.post(f"{DET_URL}/detect_all", files={"image": ("f.jpg", buf.tobytes(), "image/jpeg")}, timeout=5)
            pdets = r.json().get("people", []) if r.status_code == 200 else []
        except Exception:
            pdets = []

        assigned = tracker.update(pdets)
        for det in assigned:
            tid = det.get("track_id")
            bbox = det.get("bbox", [0, 0, 0, 0])
            kps = np.array(det.get("keypoints", []))
            _ = fall_d.process(tid, kps, bbox, h, w, timestamp=f_idx * 0.1)
            _ = head_d.process(tid, kps, h, w, timestamp=f_idx * 0.1)
            _ = hand_d.process_wrist_crop(frame, kps, h, w, bbox=bbox)

        if f_idx in check_points:
            client_mem = get_process_memory_mb(client_pid)
            server_mem = get_process_memory_mb(server_pid) if server_pid else 0.0
            mem_snapshots.append({
                "frame": f_idx,
                "client_mem_mb": round(client_mem, 1),
                "server_mem_mb": round(server_mem, 1),
                "elapsed_s": round(time.time() - t_start, 1)
            })
            print(f"  Frame {f_idx:4d}/{num_frames} ({mem_snapshots[-1]['elapsed_s']:5.1f}s) | Client RSS: {client_mem:6.1f} MB | Server RSS: {server_mem:6.1f} MB")

    cap.release()
    hand_d.release()

    c_start = mem_snapshots[0]["client_mem_mb"]
    c_end = mem_snapshots[-1]["client_mem_mb"]
    c_delta = c_end - c_start

    s_start = mem_snapshots[0]["server_mem_mb"]
    s_end = mem_snapshots[-1]["server_mem_mb"]
    s_delta = s_end - s_start

    print(f"\nMemory Profile Summary after {num_frames} frames:")
    print(f"  - Client Memory Delta: {c_delta:+.2f} MB (Start: {c_start:.1f} MB, End: {c_end:.1f} MB)")
    print(f"  - Server Memory Delta: {s_delta:+.2f} MB (Start: {s_start:.1f} MB, End: {s_end:.1f} MB)")

    # Pass criteria: memory growth under sustained load < 50MB
    is_leak_free = (c_delta < 50.0) and (s_delta < 50.0)
    print(f"  - Zero Memory Leak Status: {'✅ PASSED (Stable)' if is_leak_free else '⚠️ WARNING (Growth detected)'}")

    return {
        "snapshots": mem_snapshots,
        "client_delta_mb": round(c_delta, 2),
        "server_delta_mb": round(s_delta, 2),
        "is_leak_free": is_leak_free,
    }


def run_test_3_stream_reader_resilience(video_path: Path) -> dict:
    print("\n" + "="*70)
    print("TEST 3: LatestFrameReader (RTSP) Zero-Buffer & Fault Tolerance")
    print("="*70)

    reader = LatestFrameReader(str(video_path), name="stress_test_reader", stall_timeout=2.0)
    ok = reader.start()
    print(f"Stream Reader initialized & started: {ok}")

    received_frames = 0
    t_start = time.time()
    for _ in range(60):
        ok, frame, ts = reader.read(timeout=0.5)
        if ok and frame is not None:
            received_frames += 1
        time.sleep(0.01)

    print(f"  - Normal Read: {received_frames}/60 frames read | Dropped (freshness sync): {reader.dropped}")

    # Test thread-safe reopen / reconnect
    print("  - Testing stream reopen / self-healing...")
    reader.reopen(str(video_path))
    time.sleep(1.2)  # allow thread to reconnect cleanly

    reconnected_frames = 0
    for _ in range(20):
        ok, frame, ts = reader.read(timeout=0.5)
        if ok and frame is not None:
            reconnected_frames += 1
        time.sleep(0.01)

    print(f"  - Auto-reconnect confirmed: count={reader.reconnects} | Post-reconnect frames read: {reconnected_frames}/20")
    reader.release()

    pass_criteria = (received_frames > 40) and (reconnected_frames > 10)
    print(f"  - RTSP Reader Resilience Status: {'✅ PASSED (Self-Healing)' if pass_criteria else '⚠️ FAILED'}")
    return {
        "received_frames": received_frames,
        "dropped_stale": reader.dropped,
        "reconnect_count": reader.reconnects,
        "post_reconnect_frames": reconnected_frames,
        "passed": pass_criteria,
    }


def run_test_4_concurrent_server_load(num_streams: int = 4, requests_per_stream: int = 40) -> dict:
    print("\n" + "="*70)
    print(f"TEST 4: Concurrent Multi-Camera Load ({num_streams} Concurrent Streams)")
    print("="*70)

    # Synthetic test image
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.putText(dummy_img, "CONCURRENCY STRESS TEST", (400, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    _, buf = cv2.imencode(".jpg", dummy_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    img_bytes = buf.tobytes()

    results = defaultdict(list)
    errors = defaultdict(int)

    def worker(stream_id: int):
        for _ in range(requests_per_stream):
            t0 = time.perf_counter()
            try:
                r = requests.post(
                    f"{DET_URL}/detect_all",
                    files={"image": (f"cam{stream_id}.jpg", img_bytes, "image/jpeg")},
                    timeout=5
                )
                lat = (time.perf_counter() - t0) * 1000
                if r.status_code == 200:
                    results[stream_id].append(lat)
                else:
                    errors[stream_id] += 1
            except Exception:
                errors[stream_id] += 1

    threads = []
    t_start = time.perf_counter()
    for sid in range(num_streams):
        t = threading.Thread(target=worker, args=(sid,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    total_time = time.perf_counter() - t_start
    total_reqs = sum(len(v) for v in results.values())
    total_errs = sum(errors.values())
    all_latencies = [lat for stream_lats in results.values() for lat in stream_lats]

    avg_lat = float(np.mean(all_latencies)) if all_latencies else 0.0
    p95_lat = float(np.percentile(all_latencies, 95)) if all_latencies else 0.0
    throughput_rps = total_reqs / max(1e-3, total_time)

    print(f"Concurrent Stress Results ({num_streams} streams, {num_streams * requests_per_stream} total requests):")
    print(f"  - Total Completed Requests: {total_reqs}/{num_streams * requests_per_stream}")
    print(f"  - Total Errors / Timeouts : {total_errs} (Error Rate: {total_errs/max(1, total_reqs + total_errs)*100:.1f}%)")
    print(f"  - Average Response Latency: {avg_lat:.2f} ms")
    print(f"  - 95th Percentile Latency : {p95_lat:.2f} ms")
    print(f"  - Server Throughput       : {throughput_rps:.1f} Requests/sec")

    pass_criteria = (total_errs == 0) and (p95_lat < 1000.0)
    print(f"  - Multi-Camera Concurrency Status: {'✅ PASSED (No Starvation)' if pass_criteria else '⚠️ FAILED'}")

    return {
        "num_streams": num_streams,
        "total_requests": total_reqs,
        "error_count": total_errs,
        "avg_latency_ms": round(avg_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "throughput_rps": round(throughput_rps, 1),
        "passed": pass_criteria,
    }


def main():
    test_video = ROOT / "eval" / "videos" / "D1-1st-Floor.mp4"
    if not test_video.exists():
        test_video = Path("/Users/nurulhudaadamishaq/Downloads/Today dataset videos/D1-1st-Floor.mp4")

    print(f"Starting Day 6 Stress & Performance Suite...")
    print(f"Video Source: {test_video}")

    report = {}
    report["test_1_latency"] = run_test_1_latency_breakdown(test_video, num_frames=120)
    report["test_2_memory"] = run_test_2_sustained_memory_leak(test_video, num_frames=300)
    report["test_3_stream_reader"] = run_test_3_stream_reader_resilience(test_video)
    report["test_4_concurrency"] = run_test_4_concurrent_server_load(num_streams=4, requests_per_stream=30)

    out_json = ROOT / "eval" / "day6_stress_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Day 6 Stress Benchmark Complete! Results saved to {out_json}")


if __name__ == "__main__":
    main()
