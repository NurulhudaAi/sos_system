"""
realtime_demo.py  –  Live webcam / RTSP stream fall detection
"""

import cv2, time, argparse, sys
sys.path.append("../src")
from fall_detector import DangerousFallDetector, FallThresholds
import yaml
from pathlib import Path


def load_thresholds(path="configs/thresholds.yaml") -> FallThresholds:
    th = FallThresholds()
    if Path(path).exists():
        cfg = yaml.safe_load(Path(path).read_text())
        for k, v in cfg.items():
            if hasattr(th, k):
                setattr(th, k, v)
        print(f"Loaded thresholds from {path}")
    return th


def run(source=0, model="yolov8l-pose.pt", device="cpu", display=True):
    th       = load_thresholds()
    detector = DangerousFallDetector(model, th, device)

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    fps_history = []
    print("Press Q to quit")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        annotated, events = detector.process_frame(frame)

        # FPS overlay
        elapsed = time.time() - t0
        fps_history.append(1.0 / max(elapsed, 1e-4))
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)
        cv2.putText(annotated, f"FPS: {avg_fps:.1f}",
                    (annotated.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        if events:
            for ev in events:
                print(f"[FALL] id={ev['track_id']}  score={ev['danger_score']}  "
                      f"dur={ev['duration_sec']}s")

        if display:
            cv2.imshow("Fall Detector", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0",    help="0=webcam, path, or rtsp://...")
    p.add_argument("--model",  default="../runs/fall_detect/exp1/weights/best.pt")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    run(args.source, args.model, args.device)
