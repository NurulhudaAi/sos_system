# รายงานผลการสแกน Dataset วิดีโอจริงทั้ง 7 คลิป (Full CCTV Scan Report)

**บันทึกเมื่อ:** 2026-10-22  
**แหล่งที่มา:** `/Users/nurulhudaadamishaq/Downloads/Today dataset videos`  
**สตรีม & โคเดก:** CCTV 2688×1520 (4MP) & 1280×960 @ 8.0 FPS  
**จำนวนคลิปรวม:** 7 คลิป (ความยาวรวมประมาณ 29 นาที)  
**จำนวน Snapshot ที่บันทึกพร้อม Bounding Box:** 30 ภาพ (`eval/scan_reports/snapshots/`)

---

## 📊 ตารางสรุปภาพรวมรายคลิป

| # | ชื่อไฟล์วิดีโอ | ความยาว | สถานที่ | จำนวน Alert ที่พบ | แยกประเภทเหตุการณ์ | ผลการประเมินเบื้องต้น |
|---|---|---|---|---|---|---|
| 1 | `E1-Hall 2.mp4` | 0.8 นาที | โถงทางเดิน E1 (มุม 2) | **0** | - | ✅ **Clean Pass 100%**: ไม่มี False Alarm เลย |
| 2 | `D1-Stair-FL2 2.mp4` | 2.5 นาที | บันไดชั้น 2 (มุม 2) | **6** | Pose SOS: 6 | ✅ ตรวจจับคนชูสองมือขอความช่วยเหลือติด / ไม่มี False Fall |
| 3 | `D1-Stair-FL2 1.mp4` | 3.5 นาที | บันไดชั้น 2 (มุม 1) | **2** | Pose SOS: 1, Fall: 1 | ✅ พบ Pose SOS ที่ 30s และคนล้มที่บันได 145s |
| 4 | `E1-Hall 1.mp4` | 3.0 นาที | โถงทางเดิน E1 (มุม 1) | **3** | Fall: 3 | ✅ พบคนล้มในโถงทางเดิน 3 จุด (28s, 110s, 148s) |
| 5 | `D1-1st-Floor.mp4` | 1.5 นาที | บริเวณชั้น 1 | **7** | Pose SOS: 1, Fall: 6 | ⚠️ คนล้มช่วง 45s ถูกเตือนซ้ำ 3 ครั้งเพราะ Track ID สลับ |
| 6 | `C1-FL1-Lift.avi` | 7.1 นาที | หน้าลิฟต์ ชั้น 1 | **8** | Pose SOS: 4, Fall: 4 | ✅ พบ Pose SOS ที่ 136s, 138s, 210s, 215s และล้มที่ 121s, 318s–327s |
| 7 | `D1-Carpark.mp4` | 10.7 นาที | ลานจอดรถ | **4** | Pose SOS: 4 | ✅ พบ Pose SOS กลางลานจอดรถ 4 จุด (48s, 170s, 288s, 343s) |
| **รวม** | **7 คลิป** | **~29 นาที** | | **30 Alerts** | **Pose SOS: 16, Fall: 14** | |

---

## 🔍 รายละเอียดเหตุการณ์ทั้ง 30 ช็อต (Chronological Event Details)

### 1. `C1-FL1-Lift.avi` (8 Events)
* `121.3s` — **FALL** `[tid=11]` angle=90° (รูป: `fall_tid11_121s.jpg`)
* `136.3s` — **POSE SOS** `[tid=13]` held=2.5s (รูป: `pose_sos_tid13_136s.jpg`)
* `138.1s` — **POSE SOS** `[tid=12]` held=2.5s (รูป: `pose_sos_tid12_138s.jpg`)
* `210.1s` — **POSE SOS** `[tid=0]` held=2.5s (รูป: `pose_sos_tid0_210s.jpg`)
* `215.1s` — **POSE SOS** `[tid=12]` held=2.5s (รูป: `pose_sos_tid12_215s.jpg`)
* `318.1s` — **FALL** `[tid=9]` angle=4° (รูป: `fall_tid9_318s.jpg`)
* `319.1s` — **FALL** `[tid=13]` angle=8° (รูป: `fall_tid13_319s.jpg`)
* `327.4s` — **FALL** `[tid=23]` angle=7° (รูป: `fall_tid23_327s.jpg`)

### 2. `D1-1st-Floor.mp4` (7 Events)
* `17.5s` — **FALL** `[tid=2]` angle=3° (รูป: `fall_tid2_17s.jpg`)
* `45.6s` — **POSE SOS** `[tid=2]` held=2.5s (รูป: `pose_sos_tid2_45s.jpg`)
* `45.9s` — **FALL** `[tid=1]` angle=16° (รูป: `fall_tid1_45s.jpg`)
* `46.3s` — **FALL** `[tid=17]` angle=19° (รูป: `fall_tid17_46s.jpg`) *(ID หลุดจาก tid=1)*
* `48.0s` — **FALL** `[tid=18]` angle=14° (รูป: `fall_tid18_48s.jpg`) *(ID หลุดจาก tid=1)*
* `80.9s` — **FALL** `[tid=27]` angle=40° (รูป: `fall_tid27_80s.jpg`)
* `84.8s` — **FALL** `[tid=29]` angle=17° (รูป: `fall_tid29_84s.jpg`)

### 3. `D1-Carpark.mp4` (4 Events)
* `48.1s` — **POSE SOS** `[tid=11]` held=3.0s (รูป: `pose_sos_tid11_48s.jpg`)
* `170.0s` — **POSE SOS** `[tid=27]` held=2.5s (รูป: `pose_sos_tid27_170s.jpg`)
* `288.2s` — **POSE SOS** `[tid=56]` held=2.5s (รูป: `pose_sos_tid56_288s.jpg`)
* `343.9s` — **POSE SOS** `[tid=72]` held=2.5s (รูป: `pose_sos_tid72_343s.jpg`)

### 4. `D1-Stair-FL2 1.mp4` (2 Events)
* `30.4s` — **POSE SOS** `[tid=0]` held=2.8s (รูป: `pose_sos_tid0_30s.jpg`)
* `145.7s` — **FALL** `[tid=0]` angle=39° (รูป: `fall_tid0_145s.jpg`)

### 5. `D1-Stair-FL2 2.mp4` (6 Events)
* `23.4s` — **POSE SOS** `[tid=4]` held=2.5s (รูป: `pose_sos_tid4_23s.jpg`)
* `23.6s` — **POSE SOS** `[tid=3]` held=2.5s (รูป: `pose_sos_tid3_23s.jpg`)
* `28.9s` — **POSE SOS** `[tid=6]` held=2.5s (รูป: `pose_sos_tid6_28s.jpg`)
* `43.0s` — **POSE SOS** `[tid=11]` held=2.5s (รูป: `pose_sos_tid11_42s.jpg`)
* `114.3s` — **POSE SOS** `[tid=11]` held=2.8s (รูป: `pose_sos_tid11_114s.jpg`)
* `114.6s` — **POSE SOS** `[tid=3]` held=2.5s (รูป: `pose_sos_tid3_114s.jpg`)

### 6. `E1-Hall 1.mp4` (3 Events)
* `28.9s` — **FALL** `[tid=4]` angle=15° (รูป: `fall_tid4_28s.jpg`)
* `110.5s` — **FALL** `[tid=5]` angle=35° (รูป: `fall_tid5_110s.jpg`)
* `148.9s` — **FALL** `[tid=4]` angle=21° (รูป: `fall_tid4_148s.jpg`)

### 7. `E1-Hall 2.mp4` (0 Events)
* *Clean pass — ไม่มีการแจ้งเตือนหลอก*
