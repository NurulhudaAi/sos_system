#!/usr/bin/env python3
import yaml, cv2, time, torch, sys
from pathlib import Path
from ultralytics import YOLO
from detectors.fall_detector import FallDetector
from utils import preprocess

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT/"config/thresholds.yaml").read_text())

device = ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Device: {device}")

# load model
yolo = YOLO(cfg.get('general',{}).get('yolo_model','../models/yolov8n-pose.pt'))
fall_d = FallDetector(cfg.get('fall',{}))

SKIP = cfg.get('general',{}).get('frame_skip', 1)
CONF = cfg.get('general',{}).get('person_conf', 0.30)

sources = yaml.safe_load((ROOT/"sources.yaml").read_text()).get('sources', [])

results = []

for s in sources:
    path = s.get('path')
    sid = s.get('id')
    if not path:
        continue
    p = Path(path)
    if not p.exists():
        print(f"Skipping missing: {path}")
        continue
    # determine expected label from filename
    name = p.name.lower()
    if 'fall' in name:
        expected = 'real_fall'
    elif 'trip' in name:
        expected = 'stumble'
    elif 'walk' in name or 'walking' in name:
        expected = 'non_fall'
    else:
        expected = 'unknown'

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        print(f"Cannot open: {p}")
        continue
    print(f"Processing {p} | expected={expected}")

    detected = False
    detected_time = None
    frame_count = 0
    start_time = time.time()
    try:
        while True:
            ret, raw = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % SKIP != 0:
                continue
            frame = preprocess(raw, 1920, 1080)
            h,w = frame.shape[:2]
            # run pose track (single-frame tracking not needed)
            try:
                results_y = yolo.track(frame, persist=True, conf=CONF, classes=[0], verbose=False, tracker='bytetrack.yaml', device=device)
            except Exception as e:
                # fallback to detection-only
                try:
                    results_y = yolo(frame)
                except Exception as e2:
                    print(f"YOLO error: {e} / {e2}")
                    break
            if not results_y or results_y[0].boxes is None:
                continue
            boxes = results_y[0].boxes
            kpts = results_y[0].keypoints
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu())
                tid = int(boxes.id[i].cpu()) if boxes.id is not None else i
                if kpts is None or i>=len(kpts.data):
                    continue
                kp = kpts.data[i].cpu().numpy()
                fr = fall_d.process(tid, kp, bbox, h, w)
                if fr.get('is_fallen') and not fr.get('recovered_quickly'):
                    detected = True
                    detected_time = time.time() - start_time
                    break
            if detected:
                break
    finally:
        cap.release()

    results.append({
        'path': str(p),
        'expected': expected,
        'detected': detected,
        'detected_time_s': round(detected_time,2) if detected_time else None,
        'frames': frame_count
    })

# compute metrics
pos = [r for r in results if r['expected']=='real_fall']
neg = [r for r in results if r['expected'] in ('non_fall','unknown')]
ignored = [r for r in results if r['expected']=='stumble']

TP = sum(1 for r in pos if r['detected'])
FN = sum(1 for r in pos if not r['detected'])
FP = sum(1 for r in neg if r['detected'])
TN = sum(1 for r in neg if not r['detected'])

precision = TP / (TP + FP) if (TP+FP)>0 else 0.0
recall = TP / (TP + FN) if (TP+FN)>0 else 0.0

outp = ROOT / 'logs' / 'tuning_results.csv'
with open(outp, 'w') as f:
    f.write('path,expected,detected,detected_time_s,frames\n')
    for r in results:
        f.write(f"{r['path']},{r['expected']},{r['detected']},{r['detected_time_s']},{r['frames']}\n")
    f.write(f"\nTP,{TP}\nFN,{FN}\nFP,{FP}\nTN,{TN}\nprecision,{precision}\nrecall,{recall}\n")

print('--- TUNING SUMMARY ---')
print(f"Files processed: {len(results)} (pos={len(pos)} neg={len(neg)} ignored={len(ignored)})")
print(f"TP={TP} FN={FN} FP={FP} TN={TN}")
print(f"precision={precision:.3f} recall={recall:.3f}")
print(f"Detailed results written to: {outp}")
