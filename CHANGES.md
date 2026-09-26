# Fixes applied — copy these files back into your repo

| File → repo path | What changed |
|---|---|
| `main.py` → `main.py` | 1) Import `database` module. 2) `HelpRequestDispatcher(..., db=db_module)` — was missing `db=`, so `help_requests` collection was never written. 3) `ZoneManager(str(ROOT/"config/zones.yaml"), "default")` — was a bare relative path that silently disabled zone filtering when run from any other cwd. 4) Added `fall_d.cleanup_track(tid)` inside the existing track-drop cleanup loop — was missing entirely, so a reused track_id could inherit stale fall state (suppress a real fall or fake one). |
| `pipeline.py` → `pipeline.py` | Fixed `self.db.insert_sos_event(...)` → `self.db.insert_incident(...)` (the old method name doesn't exist in `database.py` and would have raised if `db=` were ever wired in). Also documented **why** `AlertDispatcher` should keep `db=None` in `main.py` — it already calls `insert_incident()` itself after `dispatch()`, so wiring `db=` here too would double-write every event. |
| `detectors/object_guardian.py` → `detectors/object_guardian.py` | Replaced the coordinate-based tracking key (`f"{class}_{int(x)}_{int(y)}"`, which reset on every pixel of jitter) with IoU matching against existing tracked objects — same technique as `SimpleTracker`. This is why `left_behind_seconds`/`theft_seconds` almost never fired before. Also added `track_id` to the alert dict (previously always null in `object_events`). |
| `config/zones.yaml` | No change needed — just confirm it's referenced via the corrected `main.py` path. |

## Resolved
- **PoseSOSDetector (`HandsOverHeadDetector`)**: Integrated into `main.py`, `pipeline.py`, and `utils.py`. Automatically triggers high-severity SOS incidents when a person holds both hands over head while standing for $\ge 5$ seconds.
- **MongoDB Snapshot Linking**: Added automatic resolution of `snapshot_url` & `snapshotUrl` in `insert_incident` (`database.py`) and verified against live MongoDB Atlas cluster.
- **Dynamic Camera Boundaries & Per-Detector Filtering**:
  - `ZoneManager` in `pipeline.py` supports freehand/polygon zones with OpenCV `cv2.pointPolygonTest` using normalized $[0.0, 1.0]$ coordinates.
  - Per-detector filtering (`detectors: ["fall", "hand_sos", "pose_sos", "object_guardian"]`) ensures each detector only monitors designated zones.
  - In-memory `ZONE_MANAGERS` registry enables hot-reloading zone definitions with 0s downtime without restarting inference.
  - `database.py`: Added `update_camera_zones(cam_id, zones)` and `get_camera_zones(cam_id)` persisting to `cctv_cameras.zones` in MongoDB Atlas.
  - `model_server.py`: Added `GET /api/camera/{cam_id}/frame` (snapshot capture for frontend drawing), `POST /api/camera/{cam_id}/zones` (zone persistence & hot-reload), and `GET /api/camera/{cam_id}/zones`.
  - Built interactive visual zone editor tool: `scripts/zone_editor.html`.

## Still open (not code, needs you to act)
- **Rotate the MongoDB Atlas password and purge `.env` from git history** with `git filter-repo` or BFG — this is the highest-priority item overall and isn't a code fix.
- **`record_only_dangerous: false`** in `thresholds.yaml` — currently overrides the code default (`True`), so even LOG-level falls get recorded/dispatched. Fine for testing, but worth revisiting before production to cut noise.
