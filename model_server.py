#!/usr/bin/env python3
"""
model_server.py

[FIX — Object Guardian zero-detection bug]
Root cause: `/detect_all` used ONLY the pose model (yolov8n-pose.pt) for
both people AND objects. Ultralytics pose models have exactly one class
(person, class_id=0), so the `else: objects.append(det)` branch could
NEVER fire — `objects` was always `[]`, so ObjectGuardian.update() never
saw anything to track (tp=0, fp=0, fn=all — confirmed via empty
raw_predictions_object.csv, only header row, zero data rows).

Fix: load a SECOND general-purpose YOLO model (COCO, 80 classes, e.g.
yolov8n.pt) purely for object detection. The pose model is still used
for people/keypoints (fall + hand_sos still need keypoints). Object
model results are merged in, skipping its own person (class 0)
detections since people already come from the pose model.

Config (config/thresholds.yaml -> general:):
    object_model: "../models/yolov8n.pt"   # optional override
    object_conf:  0.4                       # optional override
                                             # falls back to
                                             # object_guardian.min_confidence
                                             # then 0.4

If the object model file is missing, object detection is disabled with
a warning (fall/hand_sos pipeline keeps working — no hard crash).

[FIX — optional custom laying/standing pose model]
Root cause of "fall sometimes not detected on RTSP": the generic
COCO-pretrained yolov8n-pose.pt is trained overwhelmingly on standing /
walking people. Its person-detection confidence (and keypoint
confidence) commonly degrades — sometimes below `person_conf` entirely —
for someone lying flat, prone, or partially occluded. When that
happens, `_detect_people` returns NO bbox for that person at all, so
FallDetector.process() never even runs for them that frame, regardless
of how good its geometry/motion logic is downstream.

Fix: support an OPTIONAL second pose model fine-tuned specifically on
laying/standing posture (see the FallGuard notebook — trains
yolov8n-pose.pt further with a 2-class laying/standing head, still
17 keypoints). When configured and loadable, this model is used as the
PRIMARY person detector (better recall for lying poses, still gives
keypoints for FallDetector's geometry math) and its class output is
passed through as `posture_class` / `posture_conf` per person so
FallDetector can use it as a direct, geometry-independent "this is a
laying person" signal.

Config (config/thresholds.yaml -> general:):
    fall_pose_model: "../models/fallguard_yolov8pose.pt"   # optional
    fall_pose_class_map:                                    # optional override
      0: laying
      1: standing

If `fall_pose_model` is not set or the file can't be loaded, everything
falls back exactly to the previous behavior (generic pose model only,
no posture_class/posture_conf in the response) — fully backward
compatible.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import yaml, time, io
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
import socket

app = FastAPI()
ROOT = Path(__file__).resolve().parent
SNAPSHOT_ROOT = ROOT / "logs" / "snapshots"
cfg = yaml.safe_load((ROOT/"config/thresholds.yaml").read_text())
GENERAL = cfg.get('general', {})
YOLO_MODEL = GENERAL.get('yolo_model', '../models/yolov8n-pose.pt')
PERSON_CONF = GENERAL.get('person_conf', 0.30)

# [FIX] Separate general-object detection model config
# Default is the bare model name (not a relative path) so Ultralytics
# can auto-download+cache it if it's not present locally yet.
OBJECT_MODEL = GENERAL.get('object_model', '../models/yolov8n.pt')
OBJECT_CONF  = GENERAL.get(
    'object_conf',
    cfg.get('object_guardian', {}).get('min_confidence', 0.4)
)

# [NEW] Optional custom laying/standing pose model config
FALL_POSE_MODEL = GENERAL.get('fall_pose_model')  # None = disabled, use generic pose model only
FALL_POSE_CONF  = GENERAL.get('fall_pose_conf', PERSON_CONF)
FALL_POSE_CLASS_MAP = GENERAL.get('fall_pose_class_map', {0: 'laying', 1: 'standing'})
# normalize keys to int (yaml may load them as int already, but be safe)
FALL_POSE_CLASS_MAP = {int(k): v for k, v in FALL_POSE_CLASS_MAP.items()}

# [CLAHE] Contrast Limited Adaptive Histogram Equalization
# Improves detection accuracy in low-contrast / variable-lighting conditions.
# Validated in eval: report_after_clahe showed F1=1.0 for both fall and hand_sos
# with 0 false alarms, vs 815+ false alarms/hr without CLAHE.
CLAHE_CFG    = cfg.get('clahe', {})
CLAHE_ENABLED = CLAHE_CFG.get('enabled', True)
CLAHE_CLIP    = CLAHE_CFG.get('clip_limit', 2.0)
CLAHE_GRID    = tuple(CLAHE_CFG.get('tile_grid_size', [8, 8]))

# choose device
import torch
DEVICE = ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"[model-server] Device: {DEVICE}")

# load pose model once (people + keypoints — fall/hand_sos rely on this)
print(f"[model-server] Loading YOLO pose model: {YOLO_MODEL}")
yolo = YOLO(YOLO_MODEL)
print("[model-server] Pose model loaded")

# [FIX] load general-object model once (Object Guardian relies on this)
yolo_obj = None
try:
    print(f"[model-server] Loading YOLO object model: {OBJECT_MODEL}")
    yolo_obj = YOLO(OBJECT_MODEL)
    OBJ_NAMES = yolo_obj.names  # {class_id: class_name}
    print(f"[model-server] Object model loaded ({len(OBJ_NAMES)} classes)")
except Exception as e:
    OBJ_NAMES = {}
    print(f"⚠️  [model-server] Object model NOT loaded ({e}) — "
          f"Object Guardian will see zero objects until '{OBJECT_MODEL}' "
          f"is available. Fall/hand_sos detection is unaffected.")

# [NEW] load optional custom laying/standing pose model once.
# When present, this REPLACES the generic pose model as the primary
# person detector (better recall on lying/prone poses), while still
# providing 17-keypoint output for FallDetector's geometry math.
yolo_fall_pose = None
if FALL_POSE_MODEL:
    try:
        print(f"[model-server] Loading custom fall-pose model: {FALL_POSE_MODEL}")
        yolo_fall_pose = YOLO(FALL_POSE_MODEL)
        print(f"[model-server] Custom fall-pose model loaded "
              f"(classes={yolo_fall_pose.names})")
    except Exception as e:
        yolo_fall_pose = None
        print(f"⚠️  [model-server] Custom fall-pose model NOT loaded ({e}) — "
              f"falling back to generic pose model only. Set "
              f"general.fall_pose_model in thresholds.yaml once your "
              f"trained weights (e.g. FallGuard_YOLOv8Pose.pt) are in place.")

@app.get('/health')
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "object_model_loaded": yolo_obj is not None,
        "fall_pose_model_loaded": yolo_fall_pose is not None,
        "clahe_enabled": CLAHE_ENABLED,
    }

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


def _apply_clahe(frame):
    """Apply CLAHE preprocessing to improve detection in variable lighting.
    
    Converts to LAB color space, applies CLAHE to the L (lightness) channel,
    then converts back to BGR. This normalizes contrast without affecting
    color balance, improving YOLO and MediaPipe detection accuracy.
    
    Toggle via config/thresholds.yaml → clahe.enabled (default: true).
    """
    if not CLAHE_ENABLED:
        return frame
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"[model-server] CLAHE error (returning original frame): {e}")
        return frame


def _extract_pose_dets(results, h, w, with_posture=False):
    """Shared parsing for a pose-model result set → list of person dicts.

    When with_posture=True, also reads results[0].boxes.cls (the
    laying/standing class head of the custom fall-pose model) and
    attaches posture_class/posture_conf per detection.
    """
    people = []
    if not results or results[0].boxes is None:
        return people

    boxes = results[0].boxes
    kpts = getattr(results[0], 'keypoints', None)

    try:
        xy = boxes.xyxy.cpu().numpy()
    except Exception:
        try:
            xy = np.array(boxes.xyxy)
        except Exception:
            xy = []

    try:
        confs = boxes.conf.cpu().numpy().tolist()
    except Exception:
        try:
            confs = list(boxes.conf)
        except Exception:
            confs = []

    classes = None
    if with_posture:
        try:
            classes = boxes.cls.cpu().numpy().astype(int).tolist()
        except Exception:
            try:
                classes = [int(c) for c in boxes.cls]
            except Exception:
                classes = None

    kpts_all = []
    if kpts is not None and getattr(kpts, 'data', None) is not None:
        try:
            karr = kpts.data.cpu().numpy()
            for kp in karr:
                kp_list = []
                for x, y, c in kp:
                    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                        kp_list.append([float(x * w), float(y * h), float(c)])
                    else:
                        kp_list.append([float(x), float(y), float(c)])
                kpts_all.append(kp_list)
        except Exception:
            try:
                for item in kpts.data:
                    kp = np.array(item)
                    kp_list = []
                    for x, y, c in kp:
                        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                            kp_list.append([float(x * w), float(y * h), float(c)])
                        else:
                            kp_list.append([float(x), float(y), float(c)])
                    kpts_all.append(kp_list)
            except Exception:
                kpts_all = []

    for i_box, box in enumerate(xy.tolist() if hasattr(xy, 'tolist') else xy):
        x1, y1, x2, y2 = [float(v) for v in box]
        conf = float(confs[i_box]) if i_box < len(confs) else 0.0
        kp = kpts_all[i_box] if i_box < len(kpts_all) else []
        person = {'bbox': [x1, y1, x2, y2], 'conf': conf, 'keypoints': kp}
        if with_posture and classes is not None and i_box < len(classes):
            cls_id = classes[i_box]
            person['posture_class'] = FALL_POSE_CLASS_MAP.get(cls_id, str(cls_id))
            person['posture_conf'] = conf
        people.append(person)

    return people


def _detect_people(frame, h, w):
    """People detection with keypoints.

    [FIX] Uses the generic COCO pose model as the PRIMARY person detector
    (best recall for all poses/distances), then OVERLAYS posture_class/
    posture_conf from the optional custom fall-pose model by IoU matching.

    Previously the custom model REPLACED the generic model, but it has
    much lower recall (92% miss rate on rtsp1.mp4) — it was trained on a
    small laying/standing dataset and doesn't generalize to all camera
    angles/distances.  The generic model detects people reliably; the
    custom model adds posture classification on top.
    """
    # Step 1: Generic YOLO for person bboxes + keypoints (primary)
    try:
        results = yolo(frame, conf=PERSON_CONF, classes=[0])
    except Exception as e:
        print('[model-server] pose inference error', e)
        return []

    people = _extract_pose_dets(results, h, w, with_posture=False)

    # Step 2: If FallGuard model is available, run it and overlay posture
    if yolo_fall_pose is not None and people:
        try:
            fp_results = yolo_fall_pose(frame, conf=FALL_POSE_CONF)
            fp_dets = _extract_pose_dets(fp_results, h, w, with_posture=True)

            # Match FallGuard detections to generic detections by IoU
            for person in people:
                best_iou, best_fp = 0.0, None
                px1, py1, px2, py2 = person['bbox']
                for fp in fp_dets:
                    fx1, fy1, fx2, fy2 = fp['bbox']
                    ix1 = max(px1, fx1); iy1 = max(py1, fy1)
                    ix2 = min(px2, fx2); iy2 = min(py2, fy2)
                    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    area_p = max(1e-6, (px2-px1)*(py2-py1))
                    area_f = max(1e-6, (fx2-fx1)*(fy2-fy1))
                    iou = inter / (area_p + area_f - inter) if (area_p + area_f - inter) > 0 else 0
                    if iou > best_iou:
                        best_iou, best_fp = iou, fp
                if best_fp is not None and best_iou >= 0.3:
                    person['posture_class'] = best_fp.get('posture_class')
                    person['posture_conf'] = best_fp.get('posture_conf', 0.0)
        except Exception as e:
            print('[model-server] fall-pose overlay error (continuing without posture):', e)

    return people



def _detect_objects(frame):
    """[FIX] General-object detection via separate COCO model.
    Skips class_id==0 (person) since people already come from the pose
    model above — avoids double-counting people as 'objects'.
    """
    objects = []
    if yolo_obj is None:
        return objects
    try:
        results = yolo_obj(frame, conf=OBJECT_CONF)
    except Exception as e:
        print('[model-server] object inference error', e)
        return objects

    if not results or results[0].boxes is None:
        return objects

    boxes = results[0].boxes
    classes = getattr(boxes, 'cls', None)

    try:
        xy = boxes.xyxy.cpu().numpy()
    except Exception:
        try:
            xy = np.array(boxes.xyxy)
        except Exception:
            xy = []

    try:
        confs = boxes.conf.cpu().numpy().tolist()
    except Exception:
        try:
            confs = list(boxes.conf)
        except Exception:
            confs = []

    for i_box, box in enumerate(xy.tolist() if hasattr(xy, 'tolist') else xy):
        cls_id = int(classes[i_box]) if classes is not None and i_box < len(classes) else -1
        if cls_id == 0:
            continue  # person — handled by pose model, skip to avoid dup
        x1, y1, x2, y2 = [float(v) for v in box]
        conf = float(confs[i_box]) if i_box < len(confs) else 0.0
        class_name = OBJ_NAMES.get(cls_id, str(cls_id))
        objects.append({
            'bbox': [x1, y1, x2, y2],
            'conf': conf,
            'class_id': cls_id,
            'class_name': class_name,   # [FIX] was missing entirely before
            'confidence': conf,         # ObjectGuardian reads 'confidence'
        })

    return objects


@app.post('/detect_all')
async def detect_all(image: UploadFile = File(...)):
    """Detect people (pose model, w/ keypoints, + optional posture_class
    from the custom fall-pose model) and objects (general COCO model) —
    returns both. See module docstring for the fix rationale."""
    data = await image.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"people": [], "objects": []}
    h, w = frame.shape[:2]

    # [CLAHE] Apply contrast normalization before all detection
    frame = _apply_clahe(frame)

    people = _detect_people(frame, h, w)
    objects = _detect_objects(frame)  # [FIX] now actually populated

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

@app.get('/snapshot/hand/{filename}')
async def get_hand_snapshot(filename: str):
    """Serve hand SOS detection snapshot files."""
    filename = filename.strip('{}')
    if not filename.startswith('hand_sos_'):
        raise HTTPException(status_code=400, detail="Invalid hand SOS snapshot filename")
    snapshot_path = SNAPSHOT_ROOT / filename
    if not snapshot_path.exists():
        raise HTTPException(status_code=404, detail=f"Hand snapshot not found: {filename}")
    try:
        if not snapshot_path.is_relative_to(SNAPSHOT_ROOT):
            raise HTTPException(status_code=403, detail="Access denied")
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    return FileResponse(snapshot_path)

@app.get('/snapshots/hand')
async def list_hand_snapshots(limit: int = 10):
    """List recent hand SOS detection snapshots."""
    if not SNAPSHOT_ROOT.exists():
        return {"snapshots": [], "total": 0}

    hand_files = sorted(
        [f for f in SNAPSHOT_ROOT.glob('hand_sos_*.jpg')],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:limit]

    snapshots = []
    for jpg_file in hand_files:
        json_file = jpg_file.with_suffix('.json')
        snap = {"filename": jpg_file.name, "url": f"/snapshot/hand/{jpg_file.name}"}
        if json_file.exists():
            try:
                import json
                snap["metadata"] = json.loads(json_file.read_text())
            except:
                pass
        snapshots.append(snap)

    return {"snapshots": snapshots, "total": len(snapshots)}

if __name__=='__main__':
    # port8000
    uvicorn.run(app, host='127.0.0.1', port=8000, log_level='info')
