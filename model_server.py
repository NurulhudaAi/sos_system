#!/usr/bin/env python3
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import yaml, time, io
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()
ROOT = Path(__file__).resolve().parent
SNAPSHOT_ROOT = ROOT / "logs" / "snapshots"
cfg = yaml.safe_load((ROOT/"config/thresholds.yaml").read_text())
GENERAL = cfg.get('general', {})
YOLO_MODEL = GENERAL.get('yolo_model', '../models/yolov8n-pose.pt')
PERSON_CONF = GENERAL.get('person_conf', 0.30)

# choose device
import torch
DEVICE = ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"[model-server] Device: {DEVICE}")

# load model once
print(f"[model-server] Loading YOLO model: {YOLO_MODEL}")
yolo = YOLO(YOLO_MODEL)
print("[model-server] Model loaded")

@app.get('/health')
def health():
    return {"status":"ok","device":DEVICE}

@app.post('/detect')
async def detect(image: UploadFile = File(...)):
    data = await image.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"detections": []}
    h, w = frame.shape[:2]
    try:
        results = yolo(frame, conf=PERSON_CONF, classes=[0])
    except Exception as e:
        print('[model-server] inference error', e)
        return {"detections": []}
    dets = []
    if not results or results[0].boxes is None:
        return {"detections": []}
    boxes = results[0].boxes
    kpts = getattr(results[0], 'keypoints', None)
    # boxes.xyxy is an array of [x1,y1,x2,y2]
    xy = None
    try:
        xy = boxes.xyxy.cpu().numpy()
    except Exception:
        try:
            xy = np.array(boxes.xyxy)
        except Exception:
            xy = []
    confs = []
    try:
        confs = boxes.conf.cpu().numpy().tolist()
    except Exception:
        try:
            confs = list(boxes.conf)
        except Exception:
            confs = []
    # keypoints handling
    kpts_all = []
    if kpts is not None and getattr(kpts, 'data', None) is not None:
        try:
            karr = kpts.data.cpu().numpy()
            # karr: (N, K, 3)
            for kp in karr:
                # convert to pixel coords if normalized
                kp_list = []
                for x,y,c in kp:
                    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                        kp_list.append([float(x*w), float(y*h), float(c)])
                    else:
                        kp_list.append([float(x), float(y), float(c)])
                kpts_all.append(kp_list)
        except Exception:
            # fallback iterate
            try:
                for item in kpts.data:
                    kp = np.array(item)
                    kp_list=[]
                    for x,y,c in kp:
                        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                            kp_list.append([float(x*w), float(y*h), float(c)])
                        else:
                            kp_list.append([float(x), float(y), float(c)])
                    kpts_all.append(kp_list)
            except Exception:
                kpts_all = []

    for i_box, box in enumerate(xy.tolist() if hasattr(xy, 'tolist') else xy):
        x1,y1,x2,y2 = [float(v) for v in box]
        conf = float(confs[i_box]) if i_box < len(confs) else 0.0
        kp = kpts_all[i_box] if i_box < len(kpts_all) else []
        dets.append({
            'bbox':[x1,y1,x2,y2],
            'conf': conf,
            'keypoints': kp
        })
    return {"detections": dets}

@app.post('/detect_all')
async def detect_all(image: UploadFile = File(...)):
    """Detect people and objects — returns both."""
    data = await image.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"people": [], "objects": []}
    h, w = frame.shape[:2]
    try:
        results = yolo(frame, conf=PERSON_CONF)
    except Exception as e:
        print('[model-server] inference error', e)
        return {"people": [], "objects": []}

    people = []
    objects = []

    if not results or results[0].boxes is None:
        return {"people": people, "objects": objects}

    boxes = results[0].boxes
    kpts = getattr(results[0], 'keypoints', None)
    classes = getattr(boxes, 'cls', None)

    xy = None
    try:
        xy = boxes.xyxy.cpu().numpy()
    except Exception:
        try:
            xy = np.array(boxes.xyxy)
        except Exception:
            xy = []

    confs = []
    try:
        confs = boxes.conf.cpu().numpy().tolist()
    except Exception:
        try:
            confs = list(boxes.conf)
        except Exception:
            confs = []

    kpts_all = []
    if kpts is not None and getattr(kpts, 'data', None) is not None:
        try:
            karr = kpts.data.cpu().numpy()
            for kp in karr:
                kp_list = []
                for x,y,c in kp:
                    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                        kp_list.append([float(x*w), float(y*h), float(c)])
                    else:
                        kp_list.append([float(x), float(y), float(c)])
                kpts_all.append(kp_list)
        except Exception:
            try:
                for item in kpts.data:
                    kp = np.array(item)
                    kp_list=[]
                    for x,y,c in kp:
                        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                            kp_list.append([float(x*w), float(y*h), float(c)])
                        else:
                            kp_list.append([float(x), float(y), float(c)])
                    kpts_all.append(kp_list)
            except Exception:
                kpts_all = []

    for i_box, box in enumerate(xy.tolist() if hasattr(xy, 'tolist') else xy):
        x1,y1,x2,y2 = [float(v) for v in box]
        conf = float(confs[i_box]) if i_box < len(confs) else 0.0
        kp = kpts_all[i_box] if i_box < len(kpts_all) else []
        cls_id = int(classes[i_box]) if classes is not None and i_box < len(classes) else 0

        det = {'bbox':[x1,y1,x2,y2], 'conf': conf, 'keypoints': kp}

        if cls_id == 0:
            people.append(det)
        else:
            det['class_id'] = cls_id
            objects.append(det)

    return {"people": people, "objects": objects}

@app.get('/snapshot/{filename}')
async def get_snapshot(filename: str):
    """Serve snapshot files from logs/snapshots directory."""
    filename = filename.strip('{}')
    snapshot_path = SNAPSHOT_ROOT / filename
    if not snapshot_path.exists():
        raise HTTPException(status_code=404, detail=f"Snapshot not found: {filename}")
    try:
        if not snapshot_path.is_relative_to(SNAPSHOT_ROOT):
            raise HTTPException(status_code=403, detail="Access denied")
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    return FileResponse(snapshot_path)

if __name__=='__main__':
    print('[model-server] starting uvicorn on 127.0.0.1:8000')
    uvicorn.run(app, host='127.0.0.1', port=8000, log_level='info')
