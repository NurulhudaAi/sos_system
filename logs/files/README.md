# 🚨 Dangerous Fall Detector — YOLOv8-Pose + Temporal Analysis

ระบบตรวจจับการล้มที่อันตราย โดยลด False Positive จากการสะดุด/ก้มตัว

---

## Architecture Overview

```
Video Frame
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1 : YOLOv8-Pose (yolov8l-pose.pt)               │
│  → ตรวจจับ Person  +  Skeleton 17 keypoints             │
│  → Tracking ID  (ByteTracker)                           │
└────────────────────────┬────────────────────────────────┘
                         │  per-person keypoints + bbox
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2 : Physics Metrics  (real-time, per frame)      │
│                                                         │
│  ① Torso Angle     shoulder-mid → hip-mid  vs vertical  │
│  ② BBox Aspect     w/h > 0.8  = horizontal position     │
│  ③ Head-Below-Hip  nose y > hip y + margin              │
│  ④ Velocity Y      downward speed of centre-of-mass     │
│                                                         │
│  → Danger Score  ∈ [0, 1]  (weighted sum)              │
└────────────────────────┬────────────────────────────────┘
                         │  score per frame
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3 : Temporal Buffer  (anti-stumble filter)       │
│                                                         │
│  ✅ FALL  = score ≥ 0.60  for ≥ 8 consecutive frames   │
│                  ≈ 0.27 วินาที @ 30fps                  │
│                                                         │
│  ❌ STUMBLE = score spike แล้วกลับปกติใน < 8 frames    │
│     (คนสะดุดจะลุกขึ้นได้เร็ว ≈ 3-5 frames)            │
│                                                         │
│  🔕 CLEAR  = normal pose ≥ 20 frames → reset alert     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
                  ⚠️  FALL ALERT
                  (track_id, score, duration_sec, bbox)
```

---

## ทำไมถึงลด False Positive ได้?

| สถานการณ์     | Torso Angle | Aspect Ratio | Head Below Hip | Duration | Result    |
|---------------|-------------|--------------|----------------|----------|-----------|
| เดินปกติ      | < 30°       | < 0.6        | ✗              | -        | ❌ No alert |
| ก้มเก็บของ    | 40-60°      | < 0.7        | บางครั้ง       | < 5 fr   | ❌ No alert |
| สะดุด         | 50-70°      | 0.6-0.9      | บางครั้ง       | 3-6 fr   | ❌ No alert |
| นั่งลงพื้น    | > 60°       | > 0.9        | ✓              | < 8 fr   | ❌ No alert |
| **ล้มอันตราย** | **> 60°** | **> 0.9**    | **✓**          | **> 8 fr** | **✅ ALERT** |

---

## Dataset ที่แนะนำ

### Public Datasets (ฟรี)
| Dataset | Frames | ลิงค์ |
|---------|--------|-------|
| **Le2i Fall Detection** | ~191 videos | https://imvia.u-bourgogne.fr |
| **URFD** | 70 fall seq. | http://fenix.ur.edu.pl/mkepski/ds/uf.html |
| **FallAllD** | 26,420 samples | https://falldataset.com |
| **COCO (persons)** | 330K images | https://cocodataset.org |

### Label Format (YOLO Pose)
```
<class> <cx> <cy> <w> <h>  <kx1> <ky1> <conf1>  <kx2> <ky2> <conf2>  ... (×17)

class 0 = person_normal
class 1 = person_FALLING  ← dangerous fall only (ไม่รวมสะดุด!)
```

---

## การติดตั้ง

```bash
# สร้าง environment
conda create -n falldet python=3.11
conda activate falldet

# ติดตั้ง dependencies
pip install ultralytics==8.3.0
pip install albumentations==1.3.1
pip install scikit-learn matplotlib opencv-python
pip install pyyaml

# ดาวน์โหลด base model (YOLO จะ auto-download)
python -c "from ultralytics import YOLO; YOLO('yolov8l-pose.pt')"
```

---

## Workflow

### 1️⃣ เตรียม Dataset
```bash
# สร้าง pseudo-labels จากวิดีโอดิบ
python src/train.py --action pseudo

# แบ่ง train/val
python src/train.py --action split
```

### 2️⃣ Augment Fall Frames (แก้ class imbalance)
```bash
python src/augment_falls.py
```

### 3️⃣ เทรนโมเดล
```bash
# GPU (แนะนำ)
python src/train.py --action train --epochs 150 --batch 16 --device 0

# CPU (ช้ากว่า)
python src/train.py --action train --epochs 150 --batch 8 --device cpu
```

### 4️⃣ Evaluate + หา Optimal Threshold
```bash
python src/evaluate.py
# → ดู results/roc_curve.png
# → configs/thresholds.yaml จะถูก update อัตโนมัติ
```

### 5️⃣ Run Inference
```bash
# บนวิดีโอ
python src/fall_detector.py video.mp4 output.mp4 runs/fall_detect/exp1/weights/best.pt

# บน webcam  (realtime)
python scripts/realtime_demo.py --source 0
```

---

## Tuning Thresholds

แก้ใน `src/fall_detector.py` → `FallThresholds`:

```python
@dataclass
class FallThresholds:
    torso_angle_deg: float = 50.0       # ↓ ลด = จับได้มากขึ้น แต่ FP เพิ่ม
    bbox_aspect_ratio: float = 0.8      # ↓ ลด = จับได้มากขึ้น
    fall_confirm_frames: int = 8        # ↑ เพิ่ม = ลด FP มากขึ้น แต่ latency เพิ่ม
    fall_clear_frames: int = 20         # ↑ เพิ่ม = alert ดับช้าลง
    danger_score_threshold: float = 0.60 # ↑ เพิ่ม = เข้มงวดขึ้น ลด FP
```

**แนะนำ tradeoff:**
- `fall_confirm_frames = 8`  → latency ≈ 0.27s @ 30fps  (acceptable for alert)
- `danger_score_threshold = 0.65`  → ลด FP ดีมาก, ยังจับ real falls ได้ครบ
- `beta = 2.0` ใน evaluate.py → เน้น recall (จับได้ครบ) มากกว่า precision

---

## ผลลัพธ์ที่คาดหวัง (หลัง fine-tune)

| Metric           | ค่าที่คาดหวัง |
|------------------|--------------|
| Fall Recall      | > 92%        |
| Fall Precision   | > 85%        |
| False Positive Rate | < 5%      |
| Latency @ 30fps  | < 0.3s       |
| mAP50 (YOLO)     | > 0.88       |

---

## Files

```
fall_detection/
├── src/
│   ├── fall_detector.py     ← Main inference engine
│   ├── train.py             ← YOLO training + dataset prep
│   ├── augment_falls.py     ← Data augmentation for fall class
│   └── evaluate.py          ← Threshold tuning + ROC curve
├── scripts/
│   └── realtime_demo.py     ← Webcam demo
├── configs/
│   ├── fall_dataset.yaml    ← auto-generated
│   └── thresholds.yaml      ← optimal thresholds (from evaluate)
├── data/
│   ├── raw/                 ← ใส่ dataset ดิบที่นี่
│   └── processed/           ← auto-split train/val
└── results/
    ├── roc_curve.png
    └── threshold_sweep.png
```
