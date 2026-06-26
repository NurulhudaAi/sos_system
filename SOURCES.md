## 📹 Multi-Source Detection System

ระบบรองรับการรัน **หลาย cameras/streams พร้อมกัน** ผ่าน `config/sources.yaml`

### 🚀 การใช้งาน

#### 1️⃣ **ตั้งค่า sources.yaml**

```yaml
sources:
  - id: "camera_bedroom"
    type: "file"
    path: "/path/to/video.mp4"
    port: 8081
    location: "ห้องนอน"
    enabled: true

  - id: "camera_hallway"
    type: "rtsp"
    path: "rtsp://admin:password@192.168.1.100:554/stream1"
    port: 8082
    location: "ห้องโถง"
    enabled: true
```

#### 2️⃣ **รันระบบ**

```bash
python3 main.py
```

ระบบจะ:
- ✅ อ่าน `config/sources.yaml`
- ✅ กรอง sources ที่มี `enabled: true`
- ✅ สร้าง VLC process สำหรับแต่ละ source
- ✅ รัน detection พร้อมกัน

### 📊 Output

```
============================================================
🎬 Multi-Source Detection System
============================================================
Enabled: 2/4 sources
  ✓ camera_bedroom     | ห้องนอน         | port 8081
  ✓ camera_hallway     | ห้องโถง         | port 8082
============================================================

[camera_bedroom] Connected | location=ห้องนอน
[camera_hallway] Connected | location=ห้องโถง
```

### 🔧 Source Types

| Type | Example | ใช้สำหรับ |
|------|---------|-----------|
| `file` | `/path/to/video.mp4` | Local video files |
| `rtsp` | `rtsp://cam-ip/stream` | IP Cameras |
| `http` | `http://cam-ip:8000/stream` | MJPEG streams |

### ⚙️ Configuration Fields

```yaml
sources:
  - id: string              # Unique identifier
    type: "file|rtsp|http"  # Source type
    path: string            # Video path or URL
    port: int               # HTTP port for VLC (8081, 8082, ...)
    location: string        # Location name (อพย./ห้อง/ฯลฯ)
    enabled: bool           # true = run, false = skip
```

### 🔄 Parallel Processing

แต่ละ source จะ:
1. ใช้ VLC transcode เป็น HTTP stream
2. OpenCV อ่าน stream ผ่าน `http://localhost:PORT`
3. Model server ตรวจจับ fall/SOS
4. MongoDB บันทึกเหตุการณ์

```
┌─────────────────────────────────────────────────────┐
│  sources.yaml (4 enabled)                           │
└────────────────┬──────────────────────────────────┘
                 │
        ┌────────┼────────┬────────┬────────┐
        │        │        │        │        │
   [Process1] [Process2] [Process3] [Process4]
        │        │        │        │        │
     [VLC]   [VLC]   [VLC]   [VLC]
        │        │        │        │        │
    (port (port (port (port
     8081) 8082) 8083) 8084)
        │        │        │        │        │
    ┌───────────────────────────────────┐
    │   Model Server (port 8000)        │
    │   /detect_all endpoint            │
    └───────────────────────────────────┘
        │        │        │        │
    ┌──────────────────────────────────┐
    │   MongoDB Atlas                  │
    │   incidents collection           │
    └──────────────────────────────────┘
```

### 💡 Tips

**🟢 Enable/Disable sources อย่างรวดเร็ว:**
```yaml
enabled: true   # ✅ จะรัน
enabled: false  # ❌ จะข้าม
```

**🟢 ทดสอบ 1 source ที่เวลา:**
```yaml
- enabled: false  # ปิด camera 2-4
- enabled: true   # เปิด camera 1
```

**🟢 Port numbering:**
- Model server: `8000`
- Camera 1: `8081`
- Camera 2: `8082`
- Camera 3: `8083`
- etc.

### 🛑 Stop

```bash
Ctrl+C  # Graceful shutdown
```

ระบบจะ:
- ✅ Terminate VLC processes
- ✅ Close database connections
- ✅ Save logs

---

📝 **Example: เพิ่ม RTSP camera**

```yaml
sources:
  - id: "camera_bedroom"
    path: "/Users/nurulhudaadamishaq/Downloads/fall1.mp4"
    port: 8081
    location: "ห้องนอน"
    enabled: true

  - id: "camera_parking_lot"          # ← เพิ่มใหม่
    type: "rtsp"
    path: "rtsp://admin:pass@192.168.1.50:554/stream1"
    port: 8082
    location: "ที่จอดรถ"
    enabled: true
```

จากนั้น: `python3 main.py` → ทั้ง 2 cameras จะรันพร้อมกัน
