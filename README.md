# CCTV AI Emergency & Anomaly Detection System (SOS System)

ระบบตรวจจับเหตุฉุกเฉินและพฤติกรรมผิดปกติอัจฉริยะผ่านกล้องวงจรปิด (CCTV / RTSP / Video Stream) แบบเรียลไทม์ เชื่อมต่อและทำงานร่วมกับ **MongoDB Atlas** พร้อมระบบส่งแจ้งเตือนและบันทึกหลักฐานอัตโนมัติ

---

## สารบัญ
1. [ภาพรวมของโปรเจกต์ (Project Overview)](#1-ภาพรวมของโปรเจกต์-project-overview)
2. [ฟังก์ชันการตรวจจับหลัก 4 รูปแบบ (Core Detectors)](#2-ฟังก์ชันการตรวจจับหลัก-4-รูปแบบ-core-detectors)
3. [สถาปัตยกรรมของระบบ (System Architecture)](#3-สถาปัตยกรรมของระบบ-system-architecture)
4. [โครงสร้างไฟล์และคำอธิบายแต่ละโมดูล (Project Structure)](#4-โครงสร้างไฟล์และคำอธิบายแต่ละโมดูล-project-structure)
5. [การแชร์ Environment และการใช้งานร่วมกับ Docker (Docker & Team Sharing)](#5-การแชร์-environment-และการใช้งานร่วมกับ-docker-docker--team-sharing)
6. [ปัญหาและคอขวดในปัจจุบัน (Current Known Issues & Bottlenecks)](#6-ปัญหาและคอขวดในปัจจุบัน-current-known-issues--bottlenecks)
7. [Onboarding Guide: ขั้นตอนที่เพื่อนต้องทำหลัง Clone Repo](#7-onboarding-guide-ขั้นตอนที่เพื่อนต้องทำหลัง-clone-repo)
8. [เครื่องมือเสริมสำหรับทีม](#8-เครื่องมือเสริมสำหรับทีม)


---

## 1. ภาพรวมของโปรเจกต์ (Project Overview)

โปรเจกต์นี้ถูกออกแบบมาเพื่อเฝ้าระวังพื้นที่ผ่านกล้อง CCTV แบบอัตโนมัติ 24/7 โดยแปลงสตรีมวิดีโอ (RTSP หรือไฟล์วิดีโอ) เป็นข้อมูลวิเคราะห์ท่าทาง (Human Pose), ฝ่ามือ (Hand Landmarks), และวัตถุ (Objects) เพื่อประเมินเหตุการณ์วิกฤต เช่น **คนหกล้มหมดสติ**, **การส่งสัญญาณมือขอความช่วยเหลือลับ**, **การชูมือขอความช่วยเหลือ**, และ **การลืมของหรือขโมยของ** 

### จุดเด่นของระบบ
- **Multi-Source Processing:** รองรับการดึงข้อมูลกล้องหลายตัวพร้อมกันผ่าน Multi-processing หรือดึงรายชื่อกล้องที่ Active จาก MongoDB Atlas อัตโนมัติ (พร้อมถอดรหัส URL แบบ AES-256-GCM)
- **Spatial Zone Filtering:** กำหนดขอบเขตพื้นที่ตรวจจับแบบหลายเหลี่ยม (Polygon / Freehand) แยกเฉพาะตามประเภท Detector ได้ พร้อมฟีเจอร์ Hot-Reload 0s downtime
- **Dual Architecture:** แยกส่วนประมวลผล Deep Learning (Model Server) และส่วนจัดการสตรีม/ตรรกะ (Pipeline Worker) เพื่อความเสถียรและ Scale ได้ง่าย

---

## 2. ฟังก์ชันการตรวจจับหลัก 4 รูปแบบ (Core Detectors)

| Detector | วัตถุประสงค์ | วิธีการตรวจจับ / กลไก |
|---|---|---|
| **1. Fall Detector** | ตรวจจับคนหกล้ม ล้มลงกับพื้น และนอนนิ่งไม่ลุกขึ้น | วิเคราะห์ Bbox aspect ratio, มุมกระดูกสันหลัง (Spine Angle), ความเร็วการตกลงแนวดิ่ง (Velocity Spike), ระยะเวลาที่นอนราบกับพื้น (Ground Confirmation), ผสานกับโมเดลเฉพาะทาง **FallGuard Pose** |
| **2. Hand SOS Detector** | สัญญาณมือขอความช่วยเหลือสากล 3 จังหวะ (Signal for Help) | ใช้ MediaPipe Hand Landmarker ตรวจจับ 3 ขั้นตอน: (1) เปิดฝ่ามือ -> (2) พับนิ้วหัวแม่มือเข้าใน -> (3) กำ 4 นิ้วทับ พร้อมระบบ Wrist Crop และ Elevation Gate |
| **3. Hands Over Head (Pose SOS)** | ชูสองมือเหนือศีรษะขณะยืนค้าง | ใช้ YOLO Pose Keypoints ตรวจสอบข้อมือทั้ง 2 ข้างว่าอยู่สูงกว่าศีรษะ/หัวไหล่ค้างไว้ >= 2.5 วินาที (แก้ปัญหากล้องมุมกว้างระยะไกลที่มองไม่เห็นนิ้วมือ) |
| **4. Object Guardian** | ตรวจจับของถูกวางทิ้ง (Left-Behind) และของถูกขโมย (Theft) | ผูกความเป็นเจ้าของกับคน (Owner Proximity Binding) ผ่าน Wrist Keypoint / Bbox containment เมื่อเจ้าของเดินห่างเกินเวลาที่กำหนดจะแจ้งเตือน หรือหากของหายไปโดยมีผู้อื่นอยู่ใกล้จะแจ้งเตือนการโจรกรรม |

---

## 3. สถาปัตยกรรมของระบบ (System Architecture)

```
       [ RTSP CCTV Stream / Video Files / MongoDB cctv_cameras ]
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │         Pipeline Worker (main.py)   │
               │  - SimpleTracker (IOU + Centroid)   │
               │  - ZoneManager (Polygon filtering)  │
               └──────────────────┬──────────────────┘
                                  │ Frame (HTTP POST /detect_all)
                                  ▼
               ┌─────────────────────────────────────┐
               │    Model Server (model_server.py)   │
               │  - YOLOv8 Pose (Keypoints)          │
               │  - YOLOv8 Object (COCO 80 classes)  │
               │  - MediaPipe (Hand Landmarks)       │
               │  - FallGuard (Laying vs Standing)   │
               └──────────────────┬──────────────────┘
                                  │ Detections & Keypoints
                                  ▼
               ┌─────────────────────────────────────┐
               │           Event Logic Engine        │
               │  - FallDetector                     │
               │  - HandSOSDetector / PoseSOS        │
               │  - ObjectGuardian                   │
               │  - CooldownEngine (กันแจ้งเตือนซ้ำ) │
               └──────────────────┬──────────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       [ MongoDB Atlas (iam DB) ]     [ Webhook / Snapshot Server ]
       - Incidents Collection         - Real-time Alert Notification
       - Snapshot Image URL           - scripts/zone_editor.html
```

---

## 4. โครงสร้างไฟล์และคำอธิบายแต่ละโมดูล (Project Structure)

```
sos_system/
├── main.py                       # Entrypoint หลักของระบบ รันการอ่านวิดีโอ/RTSP และรันไปป์ไลน์
├── model_server.py               # FastAPI Server สำหรับรันโมเดล Deep Learning และ REST APIs
├── pipeline.py                   # CooldownEngine, ZoneManager (Polygon), และ Alert Dispatcher
├── database.py                   # โมดูลเชื่อมต่อ MongoDB Atlas (โหลดกล้อง, บันทึกเหตุการณ์, จัดการ Zone)
├── utils.py                      # ฟังก์ชันช่วยเหลือ (Preprocess, วาด Bounding Box/Skeleton, ถ่าย Snapshot)
├── vlc_stream.py                 # VLC Transcoder สำหรับแปลง RTSP เป็น MJPEG HTTP Stream
├── help_request_dispatcher.py    # ส่ง Webhook แจ้งเตือนฉุกเฉิน (Discord, Slack, REST API)
├── requirements.txt              # รายการ Python dependencies
│
├── detectors/                    # อัลกอริทึมการตรวจจับแต่ละประเภท
│   ├── fall_detector.py          # ตรวจจับการหกล้ม, คำนวณความเร็ว, Spine Angle, Ground confirmation
│   ├── hand_sos_detector.py      # ตรวจจับสัญญาณมือขอความช่วยเหลือ 3 จังหวะ (MediaPipe)
│   ├── hands_over_head_detector.py # ตรวจจับการชูสองมือเหนือศีรษะ (Pose SOS)
│   └── object_guardian.py        # ตรวจจับของลืมทิ้ง / ของถูกขโมย (Owner Binding & Distance)
│
├── config/                       # การตั้งค่าระบบ
│   ├── env_manager.py            # ตัวตรวจสอบและโหลด Environment variables
│   ├── thresholds.yaml           # ค่าเกณฑ์ Sensitivity/เวลา/โมเดล ของทุก Detector
│   ├── zones.yaml                # ค่าพิกัดพื้นที่โซนตรวจจับแบบคงที่ (Fallback)
│   └── zone_mapping.yaml         # แมปปิ้งโซนกับกล้อง
│
├── models/                       # ไฟล์น้ำหนักโมเดล Deep Learning (ไม่รวมใน Git)
│   ├── yolov8m-pose.pt           # YOLOv8 Medium Pose สำหรับตรวจจับคนและ 17 Keypoints
│   ├── yolov8n-pose.pt           # YOLOv8 Nano Pose (ทางเลือกสำหรับเครื่องที่ทรัพยากรจำกัด)
│   ├── yolov8n.pt                # YOLOv8 Nano Detect วัตถุทั่วไป 80 คลาส (สำหรับ Object Guardian)
│   ├── FallGuard_YOLOv8Pose.pt   # โมเดล Fine-tuned แยกท่า Laying / Standing
│   └── hand_landmarker.task      # MediaPipe Model สำหรับสกัดพิกัดข้อนิ้วมือ 21 จุด
│
├── scripts/                      # สคริปต์เสริมและเครื่องมือทดสอบ
│   ├── zone_editor.html          # เครื่องมือวาดขอบเขต Zone (Polygon) บนหน้าเว็บแบบ Interactive
│   ├── frontend_preview.html     # หน้าพรีวิวภาพกล้องและการตรวจจับแบบจำลอง
│   ├── test_object_guardian_steps1_2.py # Unit test สำหรับ Object Guardian
│   ├── test_detect_and_save.py   # ทดสอบยิงเฟรมเข้า Model Server และบันทึกผล
│   └── test_zones.py             # ทดสอบระบบ Zone Filtering
│
├── eval/                         # เครื่องมือประเมินผลและ Debugging
│   ├── debug_rtsp_skeleton.py    # สคริปต์สตรีม RTSP พร้อมวาด Skeleton แสดงผลเรียลไทม์
│   ├── eval_harness.py           # รันประเมินความแม่นยำเทียบกับ Label
│   └── live_full_rtsp.py         # สคริปต์ทดสอบ RTSP แบบ Standalone
│
├── alerts/                       # โฟลเดอร์เก็บภาพ Snapshot เหตุการณ์ที่ตรวจพบ
├── logs/                         # โฟลเดอร์เก็บล็อกระบบและไฟล์ฐานข้อมูลชั่วคราว
├── Dockerfile                    # ตัวกำหนด Image สำหรับสร้างคอนเทนเนอร์ Docker
├── docker-compose.yml            # ตัวกำหนด Service (model-server + pipeline)
├── .dockerignore                 # รายการไฟล์ที่ไม่รวมในการ Build Docker Image
├── .env.example                  # เทมเพลต Environment Variables สำหรับแชร์ในทีม
└── CHANGES.md                    # บันทึกประวัติการแก้ไขและผลการปรับปรุงระบบ
```

---

## 5. การแชร์ Environment และการใช้งานร่วมกับ Docker (Docker & Team Sharing)

### คำถาม: ควรทำ Docker เชื่อมกับ Docker เพื่อแชร์ Environment กับเพื่อนไหม?

**คำตอบคือ: ควรทำอย่างยิ่งครับ** เพราะโปรเจกต์นี้มี Dependency ซับซ้อนมาก:
1. **แก้ปัญหาความเข้ากันไม่ได้ของระบบ**: การติดตั้ง `opencv`, `mediapipe`, `torch`, `torchvision`, และโดยเฉพาะ `lap` (Linear Assignment Problem ซึ่งต้องใช้ C++ compiler ในการคอมไพล์) มักเจอปัญหาบน OS ต่างกัน (macOS Apple Silicon vs Windows vs Ubuntu)
2. **การแชร์ `.env` ที่ปลอดภัย**: ในทีม **ห้ามนำไฟล์ `.env` จริงที่มีรหัสผ่านขึ้น Git เด็ดขาด** ให้ใช้ไฟล์ `.env.example` แทน แล้วให้เพื่อนก็อปปี้ไปสร้างเป็น `.env` ของตนเอง

---

### ขั้นตอนการแชร์ Environment ให้เพื่อนร่วมทีม

1. **สร้างไฟล์ `.env` ของตัวเองจากเทมเพลต**:
   ```bash
   cp .env.example .env
   ```
2. **แก้ไขค่าใน `.env`**:
   กรอกค่า `MONGODB_URI`, `RTSP_SECRET`, และ `HELP_WEBHOOK_URL` (ส่งให้เพื่อนผ่านช่องทางสื่อสารที่ปลอดภัย เช่น 1Password, Slack Private Message หรือช่องทางส่วนตัว)
3. **เตรียมไฟล์โมเดลในโฟลเดอร์ `models/`**:
   เนื่องจากไฟล์ `.pt` และ `.task` มีขนาดใหญ่และอยู่ใน `.gitignore` ให้เพื่อนดาวน์โหลดหรือคัดลอกไฟล์โมเดลมาวางไว้ที่ `models/` ให้ครบ:
   - `yolov8m-pose.pt` หรือ `yolov8n-pose.pt`
   - `yolov8n.pt`
   - `FallGuard_YOLOv8Pose.pt` (ถ้าใช้งาน)
   - `hand_landmarker.task`

---

### วิธีการรันด้วย Docker Compose

ในโปรเจกต์ได้จัดเตรียม `docker-compose.yml` แยกเป็น 2 Services หลัก:
- `model-server`: รัน FastAPI โมเดลที่พอร์ต `8000`
- `pipeline`: รัน Main Process ตรวจจับกล้องและส่งเหตุการณ์เข้า Database

```bash
# 1. Build และ Start ทั้งระบบในโหมด Background
docker compose up -d --build

# 2. ดู Log การทำงานแบบเรียลไทม์
docker compose logs -f

# 3. ดู Log เฉพาะตัวประมวลผลโมเดล
docker compose logs -f model-server

# 4. หยุดการทำงาน
docker compose down
```

> **ข้อสังเกตเรื่อง GPU ใน Docker:**
> - หากเพื่อนใช้ **Linux หรือ Windows (WSL2)** ที่มี **NVIDIA GPU**: สามารถเปิดใช้งาน CUDA ในคอนเทนเนอร์เพื่อความเร็วสูงสุด
> - หากเพื่อนใช้ **macOS (Apple Silicon / M1/M2/M3)**: Docker บน Mac ไม่รองรับการ Pass-through Apple Silicon GPU (MPS) เข้าคอนเทนเนอร์ ระบบจะรันในโหมด CPU (ซึ่งรันได้ปกติ แต่อาจช้ากว่ารันแบบ Local Python เล็กน้อย)

---

## 6. ปัญหาและคอขวดในปัจจุบัน (Current Known Issues & Bottlenecks)

จากการตรวจสอบระบบในปัจจุบัน พบจุดที่ต้องระวังและพัฒนาต่อดังนี้:

### 1. ความปลอดภัยของ Credentials ใน Git History (ความสำคัญสูงสุด)
- **ปัญหา:** ในอดีตอาจมีการบันทึกค่า `MONGODB_URI` หรือ Password ลงใน Git Commit
- **แนวทางแก้ไข:** แนะนำให้หมุนเวียน (Rotate) รหัสผ่านของ MongoDB Atlas User ทันที และใช้เครื่องมือเช่น `git-filter-repo` หรือ BFG Repo-Cleaner เพื่อล้างประวัติไฟล์ `.env` เก่าออกจาก Git

### 2. การจัดการไฟล์โมเดลขนาดใหญ่ (Model Weights Distribution)
- **ปัญหา:** โฟลเดอร์ `models/*.pt` ถูกใส่ไว้ใน `.gitignore` ทำให้เมื่อเพื่อนโคลนโปรเจกต์ไป จะไม่มีไฟล์โมเดล หากรันทันทีอาจเกิด Error หาโมเดลไม่เจอ
- **แนวทางแก้ไข:** ทำสคริปต์ดาวน์โหลดโมเดลอัตโนมัติ (เช่น `scripts/download_models.sh`) หรือฝากไฟล์โมเดลไว้บน Cloud Storage / Hugging Face / Google Drive

### 3. RTSP Frame Drop และการหลุดของ Track ID ขณะคนล้ม
- **ปัญหา:** ขณะคนล้ม ลักษณะ Bounding Box จะเปลี่ยนรูปอย่างรวดเร็วมาก (จากทรงสูงแคบ -> ทรงกว้างเตี้ย) ทำให้ค่า IOU ระหว่างเฟรมดิ่งลง ยิ่งหากสตรีมผ่าน RTSP แล้วเกิด Frame Drop หรือ Network Jitter ตัว Tracker แบบเดิมอาจคิดว่าเป็น "คนใหม่" ทำให้ Track ID เปลี่ยน และเวลานับการล้มสะสมถูก Reset
- **สถานะปัจจุบัน:** ได้ทำการแก้ไขเบื้องต้นด้วย **Two-Pass Tracker (IOU + Centroid Distance Fallback)** ใน `main.py` แล้ว แต่อาจต้องคอยสังเกตหากเจอกล้องที่เน็ตไม่เสถียรมาก

### 4. ภาระการประมวลผล (Compute Bottleneck) เมื่อเปิดกล้องหลายตัว
- **ปัญหา:** การประมวลผลพร้อมกัน 3 โมเดล (YOLO Pose + YOLO Object + MediaPipe Hand) ตลอดทุกเฟรมบนกล้องหลายตัว จะกิน CPU/GPU สูงมาก
- **แนวทางปรับแต่ง:**
  - ปรับ `frame_skip: 2` หรือ `3` ใน `config/thresholds.yaml` เพื่อลดภาระ
  - เลือกรันโมเดล `yolov8n-pose.pt` แทน `yolov8m-pose.pt` สำหรับกล้องที่ไม่ต้องการความละเอียดสูงมาก

### 5. ข้อจำกัดระยะของสัญญาณมือ (Hand SOS Range Limitation)
- **ปัญหา:** MediaPipe Hand Landmark ต้องการภาพมือที่มีความละเอียดชัดเจน หากกล้อง CCTV ติดตั้งมุมสูงหรือคนอยู่ห่างจากกล้องเกินไป Hand Landmark จะจับนิ้วไม่ได้
- **สถานะปัจจุบัน:** ได้เพิ่มโมดูล **Hands Over Head (Pose SOS)** เข้ามา เพื่อให้คนสามารถชู 2 มือเหนือศีรษะขอความช่วยเหลือแทนได้ แม้จะอยู่ไกลจากกล้อง

### 6. ค่าคอนฟิก `record_only_dangerous`
- **ปัญหา:** ปัจจุบันใน `config/thresholds.yaml` ตั้งค่า `record_only_dangerous: true` ซึ่งจะบันทึกเฉพาะเหตุการณ์ระดับวิกฤต (เช่น ล้มนานเกินเวลาที่กำหนด) หากทีมกำลังทดสอบระบบ อาจทำให้เข้าใจผิดว่าระบบไม่บันทึกเหตุการณ์
- **แนวทางแก้ไข:** สามารถตั้งเป็น `false` ชั่วคราวในระหว่างการทดสอบในห้องแล็บ

---

## 7. Onboarding Guide: ขั้นตอนที่เพื่อนต้องทำหลัง Clone Repo

หากเพื่อนร่วมทีมเพิ่ง clone repository มาใหม่ ให้ทำตาม **4 ขั้นตอน** ด้านล่างนี้เพื่อเริ่มใช้งานได้ทันที:

```
[1. Clone Repo] ──> [2. สร้าง .env] ──> [3. วางไฟล์โมเดล] ──> [4. เลือกรัน (Docker / Local)]
```

### Checklist สรุปสิ่งที่เพื่อนต้องมี
- [ ] ติดตั้ง Docker Desktop หรือ Python 3.10 / 3.11
- [ ] ไฟล์ `.env` (ขอค่า `MONGODB_URI` และ `RTSP_SECRET` จากเจ้าของโปรเจกต์)
- [ ] ไฟล์โมเดลในโฟลเดอร์ `models/` (โดยเฉพาะโมเดลเฉพาะกิจ `FallGuard_YOLOv8Pose.pt` ถ้ามี)

---

### Step 1: Clone Repository
```bash
git clone <URL_REPOSITORY>
cd sos_system
```

---

### Step 2: ตั้งค่า Environment (.env) (จำเป็นมาก)
เนื่องจากไฟล์ `.env` ตัวจริงไม่ถูกเก็บใน Git เพื่อความปลอดภัย เพื่อนต้องสร้างขึ้นมาใหม่จาก Template:

```bash
cp .env.example .env
```

จากนั้นเปิดไฟล์ `.env` ด้วย Editor (VS Code / Cursor / Nano) แล้วกรอกข้อมูลสำคัญ:
1. **`MONGODB_URI`**: ขอ Connection String สำหรับต่อ MongoDB Atlas จากทีม
2. **`RTSP_SECRET`**: ขอคีย์ 32 ตัวอักษรสำหรับถอดรหัส URL กล้อง CCTV
3. **`HELP_WEBHOOK_URL`**: (ถ้าต้องการทดสอบยิง Alert เข้า Discord/Slack)

---

### Step 3: ตรวจสอบไฟล์โมเดลในโฟลเดอร์ `models/`
- โมเดลพื้นฐานอย่าง **`yolov8n.pt`**, **`yolov8m-pose.pt`**, และ **`hand_landmarker.task`** ระบบและ Ultralytics จะทำการ **ดาวน์โหลดให้อัตโนมัติเมื่อรันครั้งแรก** (ต้องต่ออินเทอร์เน็ต)
- **โมเดล Fine-tuned พิเศษ (ถ้ามี):** หากต้องการใช้ฟังก์ชันแยกท่ายืน/ท่านอนระดับสูง ให้ขอไฟล์ **`FallGuard_YOLOv8Pose.pt`** จากทีมมาวางไว้ในโฟลเดอร์:
  ```
  sos_system/models/FallGuard_YOLOv8Pose.pt
  ```
  *(หากไม่มีไฟล์นี้ ระบบจะใช้โมเดล Pose มาตรฐานตรวจจับตามปกติโดยอัตโนมัติ ไม่เกิดข้อผิดพลาด)*

---

### Step 4: เลือกรันระบบ (Docker หรือ Local)

#### ทางเลือก A: รันผ่าน Docker (แนะนำที่สุด ไม่ต้องลง Library เพิ่ม)

เหมาะสำหรับเริ่มงานทันที โดยไม่ต้องกังวลเรื่อง dependency บนเครื่อง:

```bash
# 1. Build และ Start ทั้งระบบ (Model Server + Main Pipeline)
docker compose up -d --build

# 2. ดูสถานะและ Log แบบเรียลไทม์
docker compose logs -f

# 3. หากต้องการหยุดการทำงาน
docker compose down
```

> **คำแนะนำการทดสอบใน Docker:**
> หากเพื่อนยังไม่มีกล้อง RTSP จริง และต้องการทดสอบด้วยไฟล์วิดีโอ `.mp4`:
> 1. นำไฟล์วิดีโอไปวางในโฟลเดอร์ `videos/` (เช่น `videos/sample.mp4`)
> 2. สั่งรัน pipeline ด้วยคำสั่ง:
>    ```bash
>    docker compose run --rm pipeline python3 main.py --source /app/videos/sample.mp4 --loop
>    ```

---

#### ทางเลือก B: รันบนเครื่องโดยตรง (Native Python)

เหมาะสำหรับเครื่อง Mac (Apple Silicon M1/M2/M3) เพื่อใช้ GPU (MPS Acceleration) หรือการ Debugging:

```bash
# 1. สร้าง Virtual Environment และติดตั้ง Dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. เปิด Terminal ที่ 1: รัน Model Inference Server
python3 model_server.py

# 3. เปิด Terminal ที่ 2: รัน Main Detection Pipeline
# ดึงกล้องจาก MongoDB Atlas:
python3 main.py --from-db

# หรือทดสอบกับไฟล์วิดีโอในเครื่อง:
python3 main.py --source /path/to/video.mp4 --loop
```

---

### Step 5: ตรวจสอบความถูกต้องว่าระบบทำงานเรียบร้อย (Verification)

1. **เช็คสถานะ Model Server:**
   เปิด Terminal หรือ Browser ไปที่:
   ```bash
   curl http://localhost:8000/health
   # ควรตอบกลับ: {"status":"ok","models_loaded":{...}}
   ```
2. **เปิดหน้าทดสอบโซนและพรีวิว:**
   - ดับเบิ้ลคลิกเปิดไฟล์ `scripts/frontend_preview.html` ใน Google Chrome เพื่อดูการจำลอง Bounding Box
   - เปิดไฟล์ `scripts/zone_editor.html` เพื่อทดลองวาด Polygon โซนตรวจจับ

---

## 8. เครื่องมือเสริมสำหรับทีม

- **Zone Editor Web Tool:** เปิดไฟล์ `scripts/zone_editor.html` บน Browser เพื่อทดลองวาดและปรับแต่งพื้นที่พิกัด Zone ของกล้องแต่ละตัว
- **Frontend Preview:** เปิดไฟล์ `scripts/frontend_preview.html` เพื่อดูตัวอย่างการแสดงผล Bounding Box และจำลองการแจ้งเตือน
- **Health Check API:** ตรวจสอบความพร้อมของ Model Server ได้ที่ `http://localhost:8000/health`
