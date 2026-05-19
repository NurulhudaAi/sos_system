# 🚀 Quick Start Guide — SOS Detection System

## ⚠️ Prerequisites

ให้ตรวจสอบว่าระบบ**พร้อม** ก่อนรัน:

### 1️⃣ **Model Server (port 8000)**
```bash
python3 model_server.py &
```

ตรวจสอบ:
```bash
curl http://127.0.0.1:8000/health
# Expected: {"status":"ok","device":"mps"}
```

### 2️⃣ **MongoDB Connection**
```bash
python3 test_mongodb.py
```

ถ้าล้มเหลว → ดู [Database Configuration](#database-configuration)

### 3️⃣ **VLC Installation**
```bash
# Mac
brew install vlc

# Verify
/Applications/VLC.app/Contents/MacOS/VLC --version
```

---

## 🎬 Running the System

### Option A: Single Source (File or RTSP)
```bash
python3 main.py /path/to/video.mp4 "Location Name"
```

Example:
```bash
python3 main.py ~/Downloads/fall1.mp4 "ห้องนอน"
```

### Option B: Multi-Source (from sources.yaml)
```bash
python3 main.py
```

จะอ่าน `config/sources.yaml` และรัน enabled sources ทั้งหมด

---

## 📋 Configuration

### sources.yaml
```yaml
sources:
  - id: "camera_bedroom"
    type: "file"
    path: "/path/to/video.mp4"
    port: 8081
    location: "ห้องนอน"
    enabled: true
```

Fields:
- `id`: Unique identifier
- `type`: `file` | `rtsp` | `http`
- `path`: Video path or URL
- `port`: HTTP port for VLC stream
- `location`: Display name
- `enabled`: true = run, false = skip

---

## 🔧 Troubleshooting

### ❌ "VLC หยุดทำงานก่อนกำหนด"

**Step 1: Test VLC directly**
```bash
python3 test_vlc.py
```

**Step 2: Check video file**
```bash
# Verify file exists and is readable
ls -lah /path/to/video.mp4

# Test with VLC directly
vlc /path/to/video.mp4 --play-and-exit
```

**Step 3: Check port availability**
```bash
lsof -i :8081
# Should be empty (port free)

pkill -9 VLC  # Kill lingering VLC processes
```

**Step 4: Increase VLC wait time**

Edit `main.py` line 127:
```python
url = vlc_mgr.start(wait=5.0)  # Increase from 3.0 to 5.0
```

### ❌ MongoDB "bad auth" Error

See [Database Configuration](#database-configuration) section

---

## 📊 Understanding Output

```
============================================================
🎬 Multi-Source Detection System
============================================================
Enabled: 2/3 sources
  ✓ camera_bedroom     | ห้องนอน         | port 8081
  ✓ camera_hallway     | ห้องโถง         | port 8082
============================================================

[camera_bedroom] Connected | location=ห้องนอน
[camera_hallway] Connected | location=ห้องโถง
```

✅ **Green** = Running normally
⚠️ **Yellow** = Warning (e.g., MongoDB connection issue)
❌ **Red** = Error (VLC failed, file not found, etc.)

---

## 🛑 Stopping

```bash
Ctrl+C  # Graceful shutdown
```

ระบบจะ:
- ✅ Terminate VLC processes
- ✅ Close database connections
- ✅ Save logs

---

## 🧪 Test Scripts

### Test VLC Streaming
```bash
python3 test_vlc.py
```

### Test MongoDB Connection
```bash
python3 test_mongodb.py
```

### Test Single Source
```bash
python3 run_single_source.py
```

---

## 📝 Database Configuration

### MongoDB Atlas Setup

1. **Get Connection String**
   - MongoDB Atlas → Databases → Connect
   - Copy "Python" version (pymongo)

2. **Update .env**
   ```
   MONGODB_URI=mongodb+srv://cctv:PASSWORD@cluster0.xxxxx.mongodb.net/
   MONGODB_DB_NAME=iam
   ```

3. **Add IP to Whitelist**
   - MongoDB Atlas → Security → Network Access
   - Add Current IP (or 0.0.0.0/0 for development)

4. **Test Connection**
   ```bash
   python3 test_mongodb.py
   ```

---

## 📈 Monitoring

### View Running Processes
```bash
ps aux | grep -E "python|VLC" | grep -v grep
```

### Check Port Usage
```bash
lsof -i :8000    # Model server
lsof -i :8081    # VLC stream (camera 1)
lsof -i :8082    # VLC stream (camera 2)
```

### View Logs
```bash
tail -f logs/sos_events.log
```

---

## 💡 Tips

### 🟢 Enable/Disable Source Quickly
Edit `sources.yaml`:
```yaml
- enabled: true   # ✅ Run
- enabled: false  # ❌ Skip
```

### 🟢 Test with Small File
```bash
python3 main.py ~/Downloads/test_10sec.mp4 "Testing"
```

### 🟢 Run Multiple Cameras
Add to `sources.yaml` then:
```bash
python3 main.py
```

---

## 📚 Additional Resources

- `SOURCES.md` — Multi-source setup guide
- `config/sources.yaml` — Source configuration
- `test_vlc.py` — VLC debugging
- `test_mongodb.py` — Database debugging

