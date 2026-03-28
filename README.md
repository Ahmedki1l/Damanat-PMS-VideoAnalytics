# Damanat PMS Video Analytics

CPU-optimized parking management system with real-time vehicle detection, slot occupancy tracking, and multi-camera support.

## Overview

This system processes live RTSP camera feeds to detect vehicles, determine parking slot occupancy, and generate structured events for slot status changes. Designed for deployment on low-power, CPU-only hardware.

**Key Features:**
- 🚗 Real-time vehicle detection using YOLO11 nano
- 📍 Polygon-based parking slot definition with interactive drawing tool
- 🔄 Per-slot state machine (VACANT → ENTERING → OCCUPIED → LEAVING)
- 📷 12-camera support across 2 parking floors (B1 & B2)
- ⚡ Round-robin processing at ~1 FPS per camera on a single CPU
- 📊 Structured JSON event output for integration

---

## Quick Start

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

docker run --name damanat-mysql -e MYSQL_ROOT_PASSWORD=password -e MYSQL_DATABASE=damanat_pms -p 3306:3306 -d mysql:8.0

pip install -r requirements.txt
```

### 2. Download YOLO Model

```bash
python setup_model.py                    # Downloads yolo11n.pt (~5 MB)
python setup_model.py --export onnx      # Optional: ONNX export for 1.5-2× speedup
python setup_model.py --export openvino  # Optional: OpenVINO for best CPU perf
```

### 3. Test Camera Connectivity

```bash
python test_cameras.py
```

### 4. Define Parking Slots

```bash
python draw_slots.py --camera CAM_04       # Single camera
python draw_slots.py --camera all          # All cameras sequentially
```

### 5. Run

```bash
# Multi-camera (all 14 cameras, round-robin)
python main.py

# Single camera with visualization
python main.py --camera CAM_04 --show

# Grid view (all cameras on one screen)
python grid_view.py
```

---

## Project Structure

```
Damanat-PMS-VideoAnalytics/
├── main.py                  # CLI entry point (multi-camera + single-camera)
├── config.yaml              # All camera & processing configuration
├── requirements.txt         # Python dependencies
├── setup_model.py           # YOLO model download & ONNX/OpenVINO export
├── draw_slots.py            # Interactive polygon drawing tool
├── grid_view.py             # Multi-camera live grid display
├── test_cameras.py          # Camera connectivity test
├── capture_snapshot.py      # Quick frame capture utility
│
├── models/                  # YOLO weight files (git-ignored)
│   └── yolo11n.pt
│
├── slots/                   # Per-camera slot polygon definitions
│   ├── b1_cam03.json
│   ├── b1_cam04.json
│   ├── ...
│   └── b2_cam14.json
│
└── src/                     # Core source code
    ├── __init__.py
    ├── config.py            # YAML config loader with typed dataclasses
    ├── camera_manager.py    # Multi-stream RTSP manager (round-robin)
    │
    ├── models/
    │   ├── slot.py          # ParkingSlot dataclass + JSON loader
    │   └── state_machine.py # 4-state machine with debounce logic
    │
    ├── detection/
    │   ├── detector.py      # YOLO inference wrapper
    │   └── tracker.py       # Ultralytics ByteTrack integration
    │
    ├── core/
    │   ├── slot_assigner.py # Vehicle-to-slot assignment logic
    │   └── engine.py        # Main pipeline orchestrator
    │
    └── events/
        └── event_bus.py     # JSON event emission + file logging
```

---

## Configuration

All settings are in `config.yaml`. Key sections:

### Cameras

Each camera entry specifies RTSP credentials and its slot polygon file:

```yaml
cameras:
  - id: CAM_04
    name: B1-PARKING
    floor: B1
    ip: "10.1.13.63"
    user: "kloudspot"
    password: "Kloud@123"
    slots_file: "slots/b1_cam04.json"
```

### Processing

```yaml
processing:
  mode: "round_robin"           # One camera at a time, cycling through all
  target_fps_per_camera: 1      # ~1 frame per second per camera
  stream_channel: 102           # 101 = main stream (4K), 102 = sub stream (720p)
```

### Detector

```yaml
detector:
  model_path: "models/yolo11n.pt"   # Supports .pt, .onnx, or OpenVINO dir
  confidence: 0.35                   # Detection confidence threshold
  classes: [2]                       # COCO class IDs — 2 = car
  imgsz: 480                         # Inference resolution (lower = faster)
```

### State Machine

```yaml
state_machine:
  confirm_enter_frames: 5    # Frames with vehicle before confirming OCCUPIED
  confirm_leave_frames: 8    # Frames without vehicle before confirming VACANT
```

---

## Camera Inventory

| Camera | Floor | IP | Slots File | Status |
|--------|-------|----|------------|--------|
| CAM_03 | B1 | 10.1.13.62 | `slots/b1_cam03.json` | ✅ |
| CAM_04 | B1 | 10.1.13.63 | `slots/b1_cam04.json` | ✅ |
| CAM_05 | B1 | 10.1.13.64 | `slots/b1_cam05.json` | ⚠️ No slots defined |
| CAM_06 | B1 | 10.1.13.65 | `slots/b1_cam06.json` | ✅ |
| CAM_07 | B1 | 10.1.13.66 | `slots/b1_cam07.json` | ✅ |
| CAM_08 | B1 | 10.1.13.67 | `slots/b1_cam08.json` | ✅ |
| CAM_09 | B2 | 10.1.13.68 | `slots/b2_cam09.json` | ✅ |
| CAM_10 | B2 | 10.1.13.69 | `slots/b2_cam10.json` | ❌ Not defined |
| CAM_11 | B2 | 10.1.13.70 | `slots/b2_cam11.json` | ✅ |
| CAM_12 | B2 | 10.1.13.71 | `slots/b2_cam12.json` | ❌ Not defined |
| CAM_13 | B2 | 10.1.13.72 | `slots/b2_cam13.json` | ❌ Not defined |
| CAM_14 | B2 | 10.1.13.73 | `slots/b2_cam14.json` | ✅ |

> **Note:** Ground floor cameras (CAM_01, CAM_02) are excluded — they are outdoor cameras unrelated to parking.

---

## Tools & Utilities

### `draw_slots.py` — Slot Polygon Drawing Tool

Interactive OpenCV tool for defining parking slot boundaries on camera views.

```bash
python draw_slots.py --camera CAM_04       # Single camera from config
python draw_slots.py --camera all          # All cameras sequentially
python draw_slots.py --image snapshot.jpg  # From saved image
```

**Modes** (toggle with keyboard):

| Key | Mode | Description |
|-----|------|-------------|
| *(default)* | **DRAW** | Left-click adds polygon corners, right-click finishes and prompts for a custom slot name |
| `e` | **EDIT** | Drag existing polygon vertices to reposition |
| `r` | **REMOVE** | Click inside a slot polygon to delete it |
| `n` | **RENAME** | Click inside a slot, then type a new name in the terminal |

Other keys: `u` = undo last point, `d` = delete last slot, `s` = save & quit, `q` = quit without saving.

### `grid_view.py` — Multi-Camera Grid Display

Shows all camera feeds on a single screen with slot polygon overlays.

```bash
python grid_view.py                        # All cameras
python grid_view.py --floor B1             # Filter by floor
python grid_view.py --cols 4               # Custom grid columns
python grid_view.py --width 1920 --height 1080
```

Controls: `f` = fullscreen, `1`–`9` = solo a camera, `q` = quit.

### `test_cameras.py` — Connectivity Test

```bash
python test_cameras.py                     # Tests all cameras from config
python test_cameras.py --channel 101       # Test main stream (4K)
```

### `setup_model.py` — Model Management

```bash
python setup_model.py                      # Download yolo11n.pt
python setup_model.py --export onnx        # Export to ONNX
python setup_model.py --export openvino    # Export to OpenVINO
python setup_model.py --model yolo11s.pt   # Use a different model
```

---

## Architecture

### Processing Pipeline (per frame)

```
RTSP Stream → Frame Capture → YOLO Detection → ByteTrack Tracking
    → Slot Assignment → State Machine Update → Event Emission
```

### State Machine

Each parking slot runs an independent state machine:

```
┌─────────┐   vehicle detected    ┌───────────┐   confirmed (5 frames)   ┌──────────┐
│  VACANT  │ ──────────────────→  │  ENTERING  │ ──────────────────────→  │ OCCUPIED  │
└─────────┘                       └───────────┘                           └──────────┘
     ↑                                  │                                      │
     │          vehicle disappeared     │        vehicle not detected           │
     │          before confirmation     │                                      │
     │ ◄────────────────────────────────┘                                      │
     │                                                                         ↓
     │   confirmed vacant (8 frames)   ┌──────────┐    vehicle gone      ┌──────────┐
     │ ◄──────────────────────────────  │  LEAVING  │ ◄─────────────────  │ OCCUPIED  │
     │                                  └──────────┘                      └──────────┘
     │                                       │
     │     vehicle re-detected               │
     └───────────────────────────────────────→┘ (back to OCCUPIED)
```

**Debounce logic** prevents flickering:
- `confirm_enter_frames` (default: 5): Vehicle must be present for 5 consecutive frames
- `confirm_leave_frames` (default: 8): Vehicle must be absent for 8 consecutive frames

### Slot Assignment

1. **Primary**: Bottom-center point of vehicle bounding box → point-in-polygon check
2. **Fallback**: Bounding box overlap with slot polygon ≥ 30%
3. **Tie-breaking**: If multiple vehicles map to the same slot, closest to centroid wins

### Multi-Camera Processing

In round-robin mode, the system processes one camera per cycle:

```
Time →  0.00s   0.08s   0.16s   ...   0.92s   1.00s
Camera: CAM_03  CAM_04  CAM_05  ...   CAM_14  CAM_03 (restart)
```

Each camera gets ~1 frame/second. A single YOLO model instance is shared across all cameras.

### Tracking (ByteTrack)

- **Kalman filter prediction** for motion-based tracking
- **IoU matching** to maintain stable track IDs across frames
- **Two-stage association**: high-confidence first, then low-confidence detections
- `persist=True` keeps tracker state between frames
- Not a visual re-ID system — uses position/motion only (CPU-efficient)

---

## Event Output

Events are printed as JSON to stdout and optionally to a log file.

### Event Types

```json
{"event": "vehicle_entering", "slot_id": "A1", "track_id": 5, "timestamp": "2026-03-15T22:05:00", "camera_id": "CAM_04", "floor": "B1"}
{"event": "vehicle_parked",   "slot_id": "A1", "track_id": 5, "timestamp": "2026-03-15T22:05:03", "camera_id": "CAM_04", "floor": "B1"}
{"event": "vehicle_leaving",  "slot_id": "A1", "track_id": 5, "timestamp": "2026-03-15T23:30:00", "camera_id": "CAM_04", "floor": "B1"}
{"event": "slot_vacant",      "slot_id": "A1", "track_id": 5, "timestamp": "2026-03-15T23:30:04", "camera_id": "CAM_04", "floor": "B1"}
```

### Status Summary

Periodic bulk status of all slots:

```json
{
  "type": "status_summary",
  "slots": [
    {"slot_id": "A1", "state": "OCCUPIED", "assigned_track_id": 5, "occupied": true, "camera_id": "CAM_04", "floor": "B1"},
    {"slot_id": "A2", "state": "VACANT", "assigned_track_id": null, "occupied": false, "camera_id": "CAM_04", "floor": "B1"}
  ]
}
```

### Log to File

Set `output.log_file` in `config.yaml`:

```yaml
output:
  log_file: "events.jsonl"
```

---

## CLI Reference

### `main.py`

```
usage: main.py [-h] [--config CONFIG] [--video VIDEO] [--camera CAMERA]
               [--show] [--show-camera SHOW_CAMERA] [--fps FPS]

Options:
  --config CONFIG          Path to YAML config (default: config.yaml)
  --video VIDEO            Single video file or RTSP URL (legacy mode)
  --camera CAMERA          Run single camera by ID (e.g., CAM_04)
  --show                   Show annotated video window
  --show-camera CAMERA     Which camera to visualize in multi-cam mode
  --fps FPS                Override target FPS
```

**Examples:**

```bash
python main.py                                    # All cameras, headless
python main.py --camera CAM_04 --show             # Single cam + visualization
python main.py --show --show-camera CAM_04        # Multi-cam, visualize one
python main.py --video parking_video.mp4 --show   # Local video file
```

---

## Performance Tuning

### Model Optimization

| Mode | Inference Time (per frame) | Setup |
|------|---------------------------|-------|
| `.pt` (PyTorch) | ~60-100ms | `python setup_model.py` |
| `.onnx` (ONNX Runtime) | ~30-50ms | `python setup_model.py --export onnx` |
| OpenVINO | ~20-35ms | `python setup_model.py --export openvino` |

### Configuration Knobs

| Parameter | Effect | Recommended |
|-----------|--------|-------------|
| `imgsz` | Lower = faster, less accurate | 480 for CPU |
| `confidence` | Higher = fewer false positives | 0.30–0.40 |
| `target_fps_per_camera` | Frames/sec per camera | 1 for 12 cams |
| `stream_channel` | 102 = 720p, 101 = 4K | 102 always |

### Hardware Requirements

| Scenario | Minimum CPU | RAM |
|----------|-------------|-----|
| 1 camera | 4 cores | 4 GB |
| 6 cameras (one floor) | 6 cores | 8 GB |
| 12 cameras (full system) | 8+ cores | 8-16 GB |

---

## Developer Guide

### Adding a New Camera

1. Add the camera entry to `cameras:` in `config.yaml`
2. Run `python test_cameras.py` to verify connectivity
3. Run `python draw_slots.py --camera CAM_XX` to define slot polygons
4. Restart `main.py`

### Modifying Slot Polygons

```bash
python draw_slots.py --camera CAM_04
# Use 'e' mode to drag points, 'r' to remove, 'n' to rename
```

### Adding Vehicle Classes

Edit `detector.classes` in `config.yaml`. COCO class IDs:
- `2` = car
- `5` = bus
- `7` = truck

```yaml
detector:
  classes: [2, 5, 7]  # car + bus + truck
```

### Slot Polygon JSON Format

Each camera's slot file is a JSON array:

```json
[
  {
    "id": "A1",
    "polygon": [[327, 245], [286, 162], [674, 119], [795, 179]],
    "label": "Slot A1"
  }
]
```

Coordinates are pixel positions in the camera's native resolution (720p for sub-stream).

### Integrating Events

Events are JSON on stdout. Pipe to any consumer:

```bash
python main.py 2>/dev/null | python your_consumer.py
python main.py 2>/dev/null | mosquitto_pub -t parking/events -l  # MQTT
```

Or set `output.log_file` in config for file-based consumption.

### Running as a Service (Windows)

Use `nssm` or a scheduled task:

```bash
nssm install DamanatPMS ".venv\Scripts\python.exe" "main.py"
nssm set DamanatPMS AppDirectory "D:\path\to\Damanat-PMS-VideoAnalytics"
nssm start DamanatPMS
```

### Running as a Service (Linux)

Create `/etc/systemd/system/damanat-pms.service`:

```ini
[Unit]
Description=Damanat PMS Video Analytics
After=network.target

[Service]
WorkingDirectory=/opt/damanat-pms
ExecStart=/opt/damanat-pms/.venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLO11 inference + built-in ByteTrack tracking |
| `opencv-python-headless` | Video I/O (no GUI overhead in production) |
| `shapely` | Polygon geometry for slot point-in-polygon checks |
| `pyyaml` | Configuration file parsing |
| `lap` | Linear assignment for ByteTrack (auto-installed by ultralytics) |

> **Note:** For visualization modes (`--show`, `grid_view.py`, `draw_slots.py`), OpenCV uses its built-in HighGUI. No additional GUI deps needed on Windows. On headless Linux, install `opencv-python` instead of `opencv-python-headless`.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: cv2` | Activate venv: `.venv\Scripts\activate` |
| `ModuleNotFoundError: lap` | Run `pip install lap` in venv |
| RTSP stream won't open | Check IP, credentials, and port 554 reachability |
| High CPU usage | Lower `imgsz` to 320, reduce cameras, or export to ONNX |
| Detection misses vehicles | Lower `confidence` to 0.25, increase `imgsz` to 640 |
| Flickering slot states | Increase `confirm_enter_frames` / `confirm_leave_frames` |
| Camera shows OFFLINE in grid | Check network, try `test_cameras.py` — auto-reconnects after 10s |

---

## Future Roadmap

- [ ] Vehicle registry (ANPR plate → slot mapping)
- [ ] REST API for querying slot status
- [ ] Color-histogram cross-camera matching
- [ ] Floor-aware spatial reasoning for vehicle routing
- [ ] Dashboard web UI
- [ ] MQTT / WebSocket event streaming
