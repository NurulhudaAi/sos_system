# SOS System - Safety & Object Surveillance

A real-time computer vision system that detects falls, SOS hand gestures, and monitors for left-behind objects. Built with Python, OpenCV, MediaPipe, and MongoDB.

## Features

✅ **Fall Detection** - Real-time fall detection using pose estimation  
✅ **SOS Hand Gesture Recognition** - Detects silent help requests via hand signals  
✅ **Object Guardian** - Identifies left-behind or stolen objects  
✅ **VLC Streaming** - Cross-platform live stream support  
✅ **MongoDB Integration** - Real-time incident logging and storage  
✅ **Structured Logging** - JSONL format event logging  

## System Requirements

- Python 3.9+
- GPU (NVIDIA recommended, CPU fallback available)
- 8GB RAM minimum
- Webcam or RTSP stream source

## Installation

### 1. Clone and Setup

```bash
cd sos_system
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the project root:

```bash
# MongoDB Configuration
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=sos_system

# Optional: Webhook for alerts
HELP_WEBHOOK_URL=https://your-webhook-url.com/alerts

# Logging
SOS_DB_PATH=logs/sos_events.db
FORCE_RECORD_ALERTS=false
```

**MongoDB Setup:**
- If using local MongoDB:
  ```bash
  MONGODB_URI=mongodb://localhost:27017
  ```
- If using MongoDB Atlas: Get your connection string from https://cloud.mongodb.com

### 4. Configuration

Edit `config/thresholds.yaml` to tune detection sensitivity:

```yaml
fall_detection:
  confidence_threshold: 0.6
  fall_velocity_threshold: 0.4

hand_sos:
  gesture_confidence: 0.8
  
object_guardian:
  sensitivity: 0.7
```

## Quick Start

### Option 1: Test with Webcam

```bash
python main.py --source 0
```

### Option 2: Test with Video File

```bash
python main.py --source path/to/video.mp4
```

### Option 3: Test with RTSP Stream

```bash
python main.py --source rtsp://camera-ip:554/stream
```

### Option 4: Single Frame Test

```bash
python test_main_single.py
```

## Testing Features

### Fall Detection Test
```bash
# Process a video with fall movements
python main.py --source logs/files/fall_detection/test_video.mp4
```

Expected output:
- Console logs showing pose detection confidence
- Fall event logged to MongoDB
- Video frame with pose skeleton drawn
- Alert saved to `logs/alerts.jsonl`

### SOS Gesture Test
Simply perform the SOS hand gesture (predefined hand pose) in front of the webcam.

System will:
- Log the SOS detection event
- Send webhook notification (if configured)
- Store incident in MongoDB

### MongoDB Connection Test
```bash
python test_mongodb.py
```

### VLC Stream Test
```bash
python test_vlc.py
```

## File Structure

```
sos_system/
├── main.py                  # Main pipeline (falls, SOS, object guardian)
├── run_single_source.py     # Single source runner
├── detectors/
│   ├── fall_detector.py     # Fall detection model
│   ├── hand_sos_detector.py # SOS gesture recognition
│   └── object_guardian.py   # Object tracking & alerts
├── config/
│   ├── env_manager.py       # Environment configuration
│   └── thresholds.yaml      # Detection thresholds
├── pipeline.py              # Alert cooldown & dispatch logic
├── database.py              # MongoDB operations
├── alert_logger.py          # Event logging (JSONL)
├── utils.py                 # Visualization & preprocessing
└── logs/
    ├── alerts.jsonl         # Event logs
    └── snapshots/           # Captured incident frames
```

## Common Issues & Fixes

### "MONGODB_URI not configured"
**Solution:** Edit `.env` with your MongoDB connection string

### "ModuleNotFoundError: No module named 'torch'"
**Solution:** `pip install torch torchvision` (may take time)

### "cv2 module not found"
**Solution:** `pip install opencv-python-headless`

### No camera detected
**Solution:** 
- Check camera permissions on Mac/Linux
- Try different source: `--source 1` or `--source 2`

### Low GPU memory
**Solution:** Reduce frame resolution in `config/thresholds.yaml`:
```yaml
frame_width: 640
frame_height: 480
```

## Logging & Debugging

### View Real-time Events
```bash
tail -f logs/alerts.jsonl
```

### View Snapshots
Captured incident frames are saved in `logs/snapshots/`

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tuning

| Setting | Impact | Trade-off |
|---------|--------|-----------|
| Lower confidence threshold | More detections | More false positives |
| Faster FPS | Real-time performance | Less accuracy |
| GPU enabled | 5-10x faster | Requires NVIDIA GPU |

## API Endpoints (FastAPI Server)

Start the server:
```bash
python -c "from model_server import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"
```

- `GET /health` - Server status
- `POST /detect` - Process frame
- `GET /incidents` - Get recent incidents from MongoDB

## Contributing & Testing

For your friend's testing:
1. Run with your own video/stream
2. Check `logs/alerts.jsonl` for detected events
3. Verify MongoDB has records with proper timestamps
4. Test all three detection modes: fall, SOS, object guardian

## Support

For issues:
1. Check `.env` configuration first
2. Review console error messages
3. Check MongoDB connectivity with `test_mongodb.py`
4. Enable DEBUG logging for detailed traces

---

**Last Updated:** 2026-05-20
