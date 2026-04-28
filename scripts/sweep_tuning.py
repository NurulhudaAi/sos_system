#!/usr/bin/env python3
import yaml, cv2, time, torch, json
from pathlib import Path
from ultralytics import YOLO
from detectors.fall_detector import FallDetector
from utils import preprocess

ROOT = Path(__file__).resolve().parents[1]
cfg_all = yaml.safe_load((ROOT/"config/thresholds.yaml").read_text())
fall_cfg_base = cfg_all.get('fall', {})

device = ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Device: {device}")

# load YOLO once
yolo = YOLO(cfg_all.get('general',{}).get('yolo_model','../models/yolov8n-pose.pt'))

SKIP = cfg_all.get('general',{}).get('frame_skip', 1)
CONF = cfg_all.get('general',{}).get('person_conf', 0.30)

sources = yaml.safe_load((ROOT/"sources.yaml").read_text()).get('sources', [])

# parameter grid (moderate size)
confirm_list = [6.0, 9.0, float(fall_cfg_base.get('confirm_seconds',12.0))]
spike_list = [0.10, float(fall_cfg_base.get('spike_thresh_norm',0.18)), 0.25]
motion_list = [0.01, float(fall_cfg_base.get('motion_thresh_norm',0.02)), 0.03]

results = []
run_idx = 0
start_all = time.time()
for confirm in confirm_list:
    for spike in spike_list:
        for motion in motion_list:
            run_idx += 1
            cfg = dict(fall_cfg_base)
            cfg['confirm_seconds'] = confirm
            cfg['spike_thresh_norm'] = spike
            cfg['motion_thresh_norm'] = motion
            fall_d = FallDetector(cfg)
            combo = {'confirm_seconds':confirm,'spike_thresh_norm':spike,'motion_thresh_norm':motion}
            print(f"Run {run_idx} combo={combo}")
            run_start = time.time()

            run_results = []
            for s in sources:
                path = s.get('path')
                if not path:
                    continue
                p = Path(path)
                if not p.exists():
                    print(f"Skipping missing: {p}")
                    continue
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
                detected = False
                frame_count = 0
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
                        try:
                            results_y = yolo.track(frame, persist=True, conf=CONF, classes=[0], verbose=False, tracker='bytetrack.yaml', device=device)
                        except Exception:
                            try:
                                results_y = yolo(frame)
                            except Exception:
                                break
                        if not results_y or results_y[0].boxes is None:
                            continue
                        boxes = results_y[0].boxes
                        kpts = results_y[0].keypoints
                        for i in range(len(boxes)):
                            bbox = boxes.xyxy[i].cpu().numpy().tolist()
                            tid = int(boxes.id[i].cpu()) if boxes.id is not None else i
                            if kpts is None or i>=len(kpts.data):
                                continue
                            kp = kpts.data[i].cpu().numpy()
                            fr = fall_d.process(tid, kp, bbox, h, w)
                            if fr.get('is_fallen') and not fr.get('recovered_quickly'):
                                detected = True
                                break
                        if detected:
                            break
                finally:
                    cap.release()

                run_results.append({'path':str(p),'expected':expected,'detected':detected})

            # compute metrics for this combo
            pos = [r for r in run_results if r['expected']=='real_fall']
            neg = [r for r in run_results if r['expected'] in ('non_fall','unknown')]
            TP = sum(1 for r in pos if r['detected'])
            FN = sum(1 for r in pos if not r['detected'])
            FP = sum(1 for r in neg if r['detected'])
            TN = sum(1 for r in neg if not r['detected'])
            precision = TP / (TP + FP) if (TP+FP)>0 else 0.0
            recall = TP / (TP + FN) if (TP+FN)>0 else 0.0
            run_time = time.time() - run_start
            res = {'combo':combo,'TP':TP,'FN':FN,'FP':FP,'TN':TN,'precision':precision,'recall':recall,'time_s':round(run_time,2)}
            print(f"Result: {res}")
            results.append(res)

# sort and write top3 by recall then precision
results_sorted = sorted(results, key=lambda r: (r['recall'], r['precision']), reverse=True)
out_csv = ROOT/'logs'/'sweep_results.csv'
out_json = ROOT/'logs'/'sweep_results.json'
with open(out_csv,'w') as f:
    f.write('confirm,spike,motion,TP,FN,FP,TN,precision,recall,time_s\n')
    for r in results:
        c = r['combo']
        f.write(f"{c['confirm_seconds']},{c['spike_thresh_norm']},{c['motion_thresh_norm']},{r['TP']},{r['FN']},{r['FP']},{r['TN']},{r['precision']},{r['recall']},{r['time_s']}\n")
with open(out_json,'w') as f:
    json.dump(results_sorted, f, indent=2)

print('--- SWEEP COMPLETE ---')
print(f"Total combos: {len(results)}. Top 3: ")
for i,r in enumerate(results_sorted[:3],1):
    print(i, r)
print(f"Full results: {out_csv} and {out_json}")
print(f"Elapsed: {round(time.time()-start_all,2)}s")
