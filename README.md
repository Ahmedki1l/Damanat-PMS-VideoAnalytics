# Damanat PMS Video Analytics

CPU-optimized parking management system with real-time vehicle detection, slot occupancy tracking, and multi-camera support.

## Overview

This system processes live RTSP camera feeds to detect vehicles, determine parking slot occupancy, and generate structured events for slot status changes. Designed for deployment on low-power, CPU-only hardware.

**Key Features:**
- 🚗 Real-time vehicle detection using YOLO11 nano
- 📍 Polygon-based parking slot definition with interactive drawing tool
- 🔄 Per-slot state machine (VACANT → ENTERING → OCCUPIED → LEAVING)
- 📷 14-camera support across Ground Floor (Street-facing) and Parking Floors (B1 & B2)
- ⚡ Round-robin processing at ~1 FPS per camera on a single CPU
- 🗺️ **Region of Interest (ROI) Masking** to exclude street traffic and noise
- 📊 Structured JSON event output for integration

---

## Quick Start

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Start the attached SQL Server database container
docker-compose up -d
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
python tests/test_cameras.py
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
├── tests/                   # Test & verification scripts
│   ├── test_cameras.py      # Camera connectivity test
│   └── ...
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
    │   └── engine.py        # Main pipeline orchestrator (supports ROI masks)
    │
    ├── tools/               # Internal utility tools
    │   └── roi_selector.py  # Interactive ROI visual selector
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

## Zoning (Parking Areas)

Zoning subdivides a floor into **areas** (aisles + ramps) so the ReID matcher
only searches cars that can physically be where the query is — cheaper *and*
more accurate. Zoning applies to **B1/B2 only**; Ground floor stays un-zoned.

### The model — three nested levels

```
floor  (B1)                         ← physical level
  └─ area  (B1-D, B1-E, B1-F, B1-RAMP)   ← a camera group within the floor
       └─ cameras  (CAM-03 … CAM-08)      ← each camera belongs to ONE area
            └─ slots / boundaries          ← drawn per camera
```

- **area** — a camera group within a floor (e.g. North/Center/South aisle, plus
  the ramp as its own area). A car has exactly one `current_area` at a time.
- **camera → area** — every B1/B2 camera is assigned **one** `area`. Its slots
  inherit that area automatically (you never tag a slot with an area).
- **boundary** — a polygon across a lane where two areas meet (e.g. a ramp
  throat). It connects exactly two areas (`area_from → area_to`). Crossing it
  moves a car between areas. Most cameras need **no** boundary — only the ones
  that physically see a transition (ramps).

Where data lives at runtime: areas → **`parking_areas`** table, boundaries →
**`boundaries`** table, slots → **`parking_slots`** table. `config.yaml` is the
**seed**: on the first run an empty `parking_areas` table is populated from it;
after that the **database is the source of truth** (edit it with the tool below).

### Setting up zones from zero — step by step

**1. Sketch the areas.** Decide how each floor splits into aisles + ramp(s) and
which cameras cover each. (One area can have several cameras; a camera belongs
to one area.)

**2. Define the areas** in `config.yaml` under a top-level `areas:` block:

```yaml
areas:
  - area_id: "B1-E"            # unique id; referenced by cameras + adjacency
    name: "B1 Center Aisle"
    floor: B1
    capacity: 30               # physical car limit (soft cap on the gallery)
    adjacency: { "B1-F": 15, "B1-D": 15 }   # {neighbor: transit_seconds}
  - area_id: "B1-RAMP"
    name: "B1 Ramp (Down from B2)"
    floor: B1
    capacity: 6
    adjacency: { "B2-RAMP": 30, "B1-F": 20 }   # the cross-floor link
  # … one entry per aisle + ramp on B1 and B2
```

- **`adjacency`** is the topology graph: list each *directly reachable*
  neighbour and the expected travel time in seconds. It gates the cross-area
  handoff (a car entering an area is only matched against cars that just left an
  adjacent area within its transit window). The only inter-floor edge is
  `B2-RAMP ↔ B1-RAMP`.
- **`capacity`** bounds how many cars an area's gallery holds.

**3. Assign each camera its area** in `config.yaml` (one line per B1/B2 camera;
Ground cameras get no `area`):

```yaml
cameras:
  - id: "CAM-05"
    name: "B1 Parking — Camera 05"
    floor: B1
    area: "B1-E"          # ← the only zoning field on a camera
    ip: "10.1.13.64"
    # …
```

**4. First run seeds the database.** Start the engine once; an empty
`parking_areas` table is seeded from the `areas:` block, then becomes
authoritative:

```bash
python main.py --config config.yaml
```

**5. Draw slots** per camera as usual (`python draw_slots.py --camera CAM-05`).
Slots need no area tag — they inherit the camera's area.

**6. Draw boundaries** only where two areas physically meet (the ramps). In
`draw_slots.py` press **`b`** to enter BOUNDARY mode, click a thin band across
the lane (perpendicular to traffic), right-click to finish, then answer the
prompts:

```
Enter boundary name (default: boundary_1): ramp_C_to_RAMP
Boundary FROM area (area_from): B2-C
Boundary TO area   (area_to):   B2-RAMP
```

Boundaries render magenta and save to the `boundaries` table. A junction where
3+ areas meet = draw one band per crossable pair.

`area_from` / `area_to` are **validated** against your configured areas (read
from the `parking_areas` table): a typo re-prompts with the list of valid area
ids, a blank entry cancels the boundary, and `area_from == area_to` is rejected.

A single boundary is **bidirectional** — the stored `area_from → area_to` is the
nominal direction, but a car crossing the *other* way is detected automatically
from its current area (e.g. a ramp used both down and up). So you draw **one**
band per gate, not two.

**7. Verify** the wiring:

```bash
python -c "from src.config import load_config; c=load_config('config.yaml'); \
print('areas:', [a.area_id for a in c.areas]); \
print('CAM-05 ->', c.area_for_camera('CAM-05')); \
print('B1-E adj:', c.adjacency_for('B1-E'))"
```

### Editing the database after the first seed

Because the DB (not YAML) is authoritative after the first run, edit areas with
the management tool — see [`tools/manage_areas.py`](#manage_areaspy--areaboundary-db-editor)
below (list / push from YAML / set a field / delete / re-seed).

### Entry ReID — line-crossing image (CAM-03)

A car enters via the ground ramp; the **ANPR server** sends the plate to
`POST /api/anpr/event`, then an **external line-crossing detector** sends the
B1-entrance frame from **CAM-03** to `POST /api/line-crossing`. The engine
**detects and crops the entering car** from that frame (it also contains parked
cars), seeds the Park_Entry ReID candidate, and binds it to the pending plate —
which confirms the car at the B1 entrance, placing it `IN_AREA(B1-A)`.

Send the **full CAM-03 frame** (not a pre-cropped car) as JSON, like the ANPR
image:

```json
POST /api/line-crossing
{
  "image_base64": "<base64 JPEG of the full CAM-03 frame, no data: prefix>",
  "camera_id": "CAM-03",     // optional, defaults to CAM-03
  "plate": "4976RZD",        // optional, the ANPR plate it follows (correlation)
  "timestamp": "2026-05-17T08:32:29"   // optional ISO, defaults to now
}
```

Response: `{ "status": "ok", "plate": ..., "cropped": <bool>, "bound": <bool>, "timestamp": ... }`
— `cropped` = a vehicle was detected & cropped; `bound` = tied to a pending ANPR
entry. The vehicle crop is done server-side (largest vehicle in the frame), so
the sender never needs to crop.

### Notes

- **Ground floor** (CAM-01/CAM-02) has no areas and is unaffected.
- **Ramps are their own area** — a car is "in" the ramp while on it; the ramp
  areas are the only link between floors.
- **Un-zoned fallback:** remove the `areas:` block (and camera `area` fields) and
  the system runs exactly as before (all-sessions matching).
- **Camera → area lives in `config.yaml`** for now; reading it from the Gateway
  cameras table is a planned follow-up.

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
| `b` | **BOUNDARY** | Draw an area-to-area crossing band (magenta); on finish, prompts for `area_from` / `area_to`. See [Zoning](#zoning-parking-areas). |
| `e` | **EDIT** | Drag existing polygon vertices to reposition |
| `r` | **REMOVE** | Click inside a slot polygon to delete it |
| `n` | **RENAME** | Click inside a slot, then type a new name in the terminal |

Other keys: `u` = undo last point, `d` = delete last slot, `s` = save & quit, `q` = quit without saving.

Slots save to the `parking_slots` table; boundaries save to the `boundaries` table.

### `manage_areas.py` — Area/Boundary DB Editor

Edits the zoning tables (`parking_areas`, `boundaries`) in the live database.
`config.yaml` only **seeds** `parking_areas` on the first run — after that the
database is authoritative, so use this tool to change areas.

```bash
python tools/manage_areas.py list             # Show areas in the DB (+ camera counts)
python tools/manage_areas.py boundaries        # Show boundaries in the DB
python tools/manage_areas.py push              # Apply config.yaml's areas: block to the DB (upsert)
python tools/manage_areas.py set --id B1-E --capacity 28              # Update one field
python tools/manage_areas.py set --id B1-E --name "B1 Center" \
    --floor B1 --capacity 30 --adjacency "B1-F:15,B1-D:15"            # Add/replace an area
python tools/manage_areas.py delete --id B1-E                        # Remove an area
python tools/manage_areas.py delete-boundary --id ramp_C_to_RAMP      # Remove a boundary
python tools/manage_areas.py reseed --yes                            # Wipe + re-seed from config.yaml
```

Typical workflow after editing `config.yaml`: `python tools/manage_areas.py push`.

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
python tests/test_cameras.py                     # Tests all cameras from config
python tests/test_cameras.py --channel 101       # Test main stream (4K)
```

### `setup_model.py` — Model Management

```bash
python setup_model.py                      # Download yolo11n.pt
python setup_model.py --export onnx        # Export to ONNX
python setup_model.py --export openvino    # Export to OpenVINO
python setup_model.py --model yolo11s.pt   # Use a different model
```

---

## Region of Interest (ROI) Masking

To ignore irrelevant areas (like street traffic), the system supports **ROI Masking**.

- **Setup:** Run `python src/tools/roi_selector.py` to draw the detection zone.
- **Visuals:** The ROI appears as a yellow "DETECTION ZONE" border in supervised mode.
- **Storage:** Saved as `"roi"` entries within the `slots/*.json` files.

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
2. Run `python tests/test_cameras.py` to verify connectivity
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
| `shapely` | Polygon geometry for slot point-in-polygon and ROI masking |
| `pyyaml` | Configuration file parsing |
| `lap` | Linear assignment for ByteTrack (auto-installed by ultralytics) |

> **Note:** For visualization modes (`--show`, `grid_view.py`, `draw_slots.py`, `roi_selector.py`), OpenCV uses its built-in HighGUI. No additional GUI deps needed on Windows. On headless Linux, install `opencv-python` instead of `opencv-python-headless`.

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
| Camera shows OFFLINE in grid | Check network, try `tests/test_cameras.py` — auto-reconnects after 10s |

---

## Future Roadmap

- [ ] Vehicle registry (ANPR plate → slot mapping)
- [ ] REST API for querying slot status
- [ ] Color-histogram cross-camera matching
- [ ] Floor-aware spatial reasoning for vehicle routing
- [ ] Dashboard web UI
- [ ] MQTT / WebSocket event streaming

---

## Vehicle Matching Cascade (Phase 1+)

The matching subsystem links a plate observed by ANPR at the gate to a vehicle parked on a floor camera. The original pipeline relied on a single ReID cosine score (OSNet via torchreid) running at ~1 s/image on CPU and was prone to wrong matches under similar-looking cars. The refactor in this branch replaces it with a **cascade** that gates expensive operations behind cheap ones and combines four independent modalities — ReID, color, body type, plate OCR — via an ensemble rule. ReID itself now runs through OpenVINO INT8 at **~3 ms/image** on CPU.

### Cascade pipeline

```
HSV pre-filter (1 ms)
       ↓
Color + Type classifiers (~3 ms — hard reject on confident disagreement)
       ↓
Quantised ReID cosine (~3 ms on top-K candidates only)
       ↓
Plate OCR cross-check (~30 ms — runs only in the marginal score band [0.40, 0.55])
       ↓
Ensemble verdict: confirm iff ReID ≥ 0.70 OR ≥2-of-N modalities agree
       ↓
(optional) Temporal voting: commit only when same plate wins ≥3 of last 5 frames
```

### One-time setup

#### 1. Install new dependencies

The new requirements are already in `requirements.txt`. After pulling this branch:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

New runtime deps: `paddleocr` + `paddlepaddle` (plate OCR), `faiss-cpu` (optional gallery index).
New build-time deps: `onnx` + `nncf` (OpenVINO export pipelines).

> The Python wheel chain for PaddleOCR is ~500 MB installed; the runtime model is only ~12-20 MB. Gate this dep behind a deployment feature flag in resource-constrained images.

#### 2. Build a calibration set and export OSNet to OpenVINO INT8

This is what unlocks the **3 ms/image** ReID latency. Requires ~300 sample crops from your facility.

```bash
# Auto-sample crops from existing vehicle_images/ into the calibration dir
python tools/build_calibration_set.py

# Export OSNet → ONNX → OpenVINO IR → INT8 quantised
python tools/export_osnet_openvino.py \
  --input-size 192x96 \
  --output-dir models/osnet_openvino_int8 \
  --calibration-dir tests/data/calibration_crops
```

Output: `models/osnet_openvino_int8/{model.xml, model.bin, metadata.yaml}`. The runtime auto-picks this backend when `MatchingConfig.use_openvino_reid=True` (default) and the IR exists.

Fallback: if you skip this step, the system falls back to torchreid on CPU (~1 s/image — usable only for development).

#### 3. (Optional) Train the color classifier

The plugin ships with a synthetic-trained fallback IR at `models/color_classifier_openvino/model.xml` (~97 % accuracy on synthetic test split, but unknown on real cars). For production accuracy, retrain on facility crops:

```bash
# Generate a labelling worklist from existing vehicle_images/
python tools/prepare_color_dataset.py --out tests/data/color_classifier/

# After humans fill in the confirmed_color column in worklist.csv:
python tools/train_color_classifier.py \
  --manifest tests/data/color_classifier/manifest.csv \
  --out models/color_classifier_openvino \
  --epochs 30
```

Target dataset size: **500 crops × 11 classes** (black/white/grey/silver/red/blue/green/yellow/brown/beige/other).

#### 4. (Optional) Train the body-type classifier

Same pattern. Default classes: sedan / SUV / hatchback / pickup / van / bus.

```bash
python tools/prepare_type_dataset.py --output-dir tests/data/type_classifier/
# Humans fill in confirmed_type column in the emitted train/val/test CSVs
python tools/train_type_classifier.py \
  --data-dir tests/data/type_classifier/ \
  --output-dir models/type_classifier_openvino \
  --epochs 30
```

Target dataset size: **500 crops × 6 classes**.

#### 5. Verify plugin loading

```bash
python -c "from src.vehicle_registry import VehicleRegistry; r = VehicleRegistry(); print(type(r.match_decision._color_classifier).__name__, type(r.match_decision._type_classifier).__name__, type(r.match_decision._plate_ocr).__name__)"
```

Expected when all four plugin IRs are present:

```
OpenVINOColorClassifier OpenVINOTypeClassifier PaddlePlateOCR
```

Any of the three falling back to `Noop*` means the corresponding model is missing — the cascade still works but loses that modality's vote.

### Configuration

Add or update the `matching:` block in `config.yaml`. All fields are optional; defaults match the values shown below.

```yaml
matching:
  # ReID thresholds (will need calibration after switching to OpenVINO INT8 —
  # see "Threshold calibration" below)
  b1_anpr: 0.47               # B1 confirmation when candidate is an ANPR crop
  b1_zone: 0.55               # B1 confirmation for Park-Entry zone crops
  b1_cross_camera: 0.43       # B1 confirmation when handing off between cameras
  global_default: 0.55        # Cross-session global search
  global_with_plate: 0.46     # Same, when the session has a confirmed plate
  global_cross_camera: 0.43   # Same, when query is on a different camera
  reattach_default: 0.52      # Track→session reattach (orphan recovery)
  reattach_cross_camera: 0.43
  legacy_color_fallback: 0.35
  color_dominant_filter: 0.45

  # Marginal band — only here does the plate OCR fire
  ocr_marginal_low: 0.40
  ocr_marginal_high: 0.55

  # ReID-solo confirm fast-path (single-modality high-confidence)
  reid_solo_confirm: 0.70

  # HSV pre-filter tolerances (tightened in Phase 2 — classifier is now primary)
  hsv_h_tol: 12.0
  hsv_s_tol: 80.0
  hsv_v_tol: 80.0

  # Ensemble rule
  ensemble_min_modalities_agree: 2

  # Temporal voting (feature-flagged — see "Production rollout" below)
  voting_enabled: false
  voting_window_frames: 5
  voting_min_agree: 3

  # OpenVINO ReID backend
  use_openvino_reid: true
  reid_input_size: [192, 96]
  reid_openvino_model_dir: "models/osnet_openvino_int8"

  # Plugin paths (auto-fall-back to Noop when missing)
  color_classifier_model: "models/color_classifier_openvino/model.xml"
  type_classifier_model: ""    # set when you retrain on real data
  plate_ocr_model: ""          # PaddleOCR auto-downloads on first use

  # Feature flags retained from the original pipeline
  use_color_filter: false
  use_lab_clahe: false
  use_multishot: false

  # FAISS gallery index (Phase 3 — only enable when gallery > ~5 000 sessions)
  use_faiss_index: false
  faiss_index_dimension: 512
  faiss_index_nlist: 8
```

The thresholds above are the defaults locked in `tests/data/matching_config_snapshot.json`. After running the calibration tool (below), replace them with the recommended block it emits.

### Running with the new pipeline

There is no new CLI — `main.py` already loads the `matching:` block and constructs the `MatchDecision` chokepoint behind the existing pipeline:

```bash
python main.py                                # all cameras, headless
python main.py --camera CAM_04 --show         # single camera with viz
```

What changed under the hood:

- `VehicleReIDMatcher` picks the OpenVINO backend automatically when the IR is present.
- `VehicleRegistry` auto-instantiates `OpenVINOColorClassifier` / `OpenVINOTypeClassifier` / `PaddlePlateOCR` when their IRs / models are found.
- All match decisions route through `MatchDecision.decide_b1` / `decide_global` / `decide_reattach` — no caller-side changes needed.
- Audit log line `MATCH_EVENT | Plate | Slot | Time | NewCost | OldCost | Flags | ...` (in the `reid_match_perf` logger) is unchanged in schema; it now also carries modality flags.

### Operations

#### Threshold calibration (do this first, before enabling anything else live)

The switch from torchreid to OpenVINO INT8 shifts the cosine distribution by ~0.06 (a known OSNet-AIN + InstanceNorm + ONNX export artefact). The default thresholds were calibrated for the old distribution — they are likely ~0.06 too high now.

After ≥1 week of production logs accumulate:

```bash
# Read-only — emits a JSON report (stdout or --output file) including a
# recommended `matching:` block.
python tools/calibrate_thresholds.py \
  --log-glob "logs/reid_match_perf*.log" \
  --db-url "$DAMANAT_DB_URL" \
  --output calibration_report.json

# Once you've reviewed the report, apply the recommended thresholds in place:
python tools/calibrate_thresholds.py \
  --log-glob "logs/reid_match_perf*.log" \
  --db-url "$DAMANAT_DB_URL" \
  --apply config.yaml
```

The tool:

1. Parses `MATCH_EVENT` log lines.
2. Joins on `plate` against the `parking_sessions` table for ground truth (skips this step gracefully if `--db-url` is absent — sweep then runs in self-consistency mode).
3. Sweeps each threshold and reports precision/recall curves.
4. Detects mean-cosine drift and recommends a uniform shift if ≥0.03.
5. Without `--apply`, prints/writes a JSON report containing the recommended `matching:` block. With `--apply config.yaml`, rewrites the `matching:` block in the named file in place.

After applying, restart `main.py`.

#### Enable temporal voting (after thresholds are calibrated)

Voting kills flicker matches — a single bad frame can no longer commit a wrong plate. Adds 200–500 ms of commit latency depending on FPS.

```yaml
matching:
  voting_enabled: true
  voting_window_frames: 3   # smaller window for snappier UX
  voting_min_agree: 2
```

For operators who tolerate slightly slower labelling, the more conservative 5/3 setting cuts false positives further.

#### Enable FAISS gallery index (only at scale)

Below ~5 000 confirmed sessions the pure-numpy fallback inside `GalleryIndex` is fast enough. Above that, enable FAISS:

```yaml
matching:
  use_faiss_index: true
  faiss_index_dimension: 512
  faiss_index_nlist: 8   # bump to 64 above 50 000 sessions
```

`pip install faiss-cpu` first if not already present.

#### Fine-tune OSNet on facility data (long-term lever)

Closes the 0.065 cosine drift gap and adapts to facility-specific lighting / camera angles. Run after ≥10 000 `MATCH_EVENT` log entries have accumulated.

```bash
python tools/finetune_osnet_facility.py \
  --match-log-glob "logs/reid_match_perf*.log" \
  --vehicle-images-dir vehicle_images \
  --db-url "$DAMANAT_DB_URL" \
  --output-dir models/osnet_finetuned \
  --epochs 15 \
  --apply-export
```

The script:

1. Mines positive pairs from confirmed entry/exit logs (same plate, different cameras).
2. Mines hard negatives from OCR-contradiction rejection logs (and below-threshold reattach attempts).
3. Triplet-loss fine-tunes OSNet-AIN with plate-stratified train/val splits.
4. Re-exports to OpenVINO INT8 at `models/osnet_finetuned_int8/`.
5. Updates `MatchingConfig.reid_openvino_model_dir` (with `--apply-export`).

Audit the emitted `finetune_report.json` before trusting val accuracy > 0.95 — easy plates can dominate.

### Verification & testing

```bash
# Matching-cascade unit + integration tests (113+ tests, runs in ~10 s)
python -m pytest tests/test_matching_config_snapshot.py tests/test_matching_e2e.py \
                 tests/test_color_classifier.py tests/test_type_classifier.py \
                 tests/test_plate_ocr.py tests/test_matching_accuracy.py \
                 tests/test_match_voter.py tests/test_calibration_tool.py \
                 tests/test_gallery_index.py tests/test_matching_e2e_with_plugins.py -v

# ReID CPU latency benchmark (asserts <40 ms median; today's reading is ~3 ms)
python -m pytest tests/test_reid_cpu_latency.py -v

# Smoke imports
python -c "from src.matching import MatchDecision, MatchVoter, GalleryIndex; print('OK')"

# Live OCR sanity check on existing crops
python tools/test_ocr_on_crops.py --input-dir vehicle_images --max-crops 50
```

### Production rollout sequence

Step-by-step from "just merged" to "fully enabled":

1. **Install deps + export OSNet to OpenVINO INT8.** Verify ReID latency drops to <40 ms median (`pytest tests/test_reid_cpu_latency.py`). The system already runs end-to-end at this point — cascade is in place but with classifier Noop fallbacks if their IRs aren't trained yet.
2. **Run for 1 week with default config.** Logs accumulate. Do not touch thresholds yet.
3. **Run `tools/calibrate_thresholds.py`** and paste its recommended block into `config.yaml`. This reconciles the OpenVINO cosine shift.
4. **Collect labelled data (D-2, D-3): 500 crops × 11 colors and 500 × 6 types.** One engineer-week of labelling work. Retrain the two classifiers.
5. **Enable shadow mode** for cascade verdict comparison against legacy: set `voting_enabled=False`, `use_faiss_index=False`, and review the `MATCH_EVENT` flag deltas in the log. Cut over when disagreement rate stabilises and reflects an improvement.
6. **Enable temporal voting** (`voting_enabled=True`, `voting_window_frames=3`, `voting_min_agree=2`). Monitor commit latency.
7. **Enable FAISS index** once `_sessions` count crosses ~5 000.
8. **Fine-tune OSNet on facility data** once you have ≥10 000 `MATCH_EVENT` rows. Re-export and point `reid_openvino_model_dir` at the new IR.

### Troubleshooting additions

| Issue | Solution |
|-------|----------|
| ReID still slow (>100 ms/image) | OpenVINO IR not exported. Run `python tools/export_osnet_openvino.py`. Confirm `models/osnet_openvino_int8/model.xml` exists. |
| `OpenVINOColorClassifier` raises `RuntimeError: model not found` | Either run `tools/train_color_classifier.py` or set `MatchingConfig.color_classifier_model: ""` to fall back to the Noop plugin. |
| `MATCH_EVENT` log line missing modality data | Check `Flags` field — `use_color_filter` / `use_lab_clahe` / `use_multishot` flags reflect the active config. |
| All matches confirm at marginal scores | Likely the post-INT8 cosine shift. Run `tools/calibrate_thresholds.py` and apply recommended thresholds. |
| Voting never commits | Buffer not filling — check that the camera FPS × `voting_window_frames` ≥ time-to-commit-budget. Try `voting_window_frames: 3`, `voting_min_agree: 2`. |
| PaddleOCR crashes on Windows with oneDNN error | Plugin already disables oneDNN (`FLAGS_use_mkldnn=0`); if you re-enable it via env var, you'll see this. Leave the default. |
| FAISS import error | `pip install faiss-cpu`. Or set `use_faiss_index: false` — `GalleryIndex` falls back to a pure-numpy brute-force search with identical correctness. |

### Retraining ReID on new facility data (Top-20 loop)

The full pipeline that takes a folder of organised plate captures and ships a fine-tuned ReID model to production is implemented as four scripts plus the existing exporter. Each step has clean inputs and outputs so any one can be re-run independently.

#### Dataset shape

Organise your facility crops as one folder per identity:

```
<source-root>/
  <PLATE-XXXX>/
    *.jpg            # any captures of this vehicle (entry, floor, etc.)
  <PLATE-YYYY>/
    *.jpg
  ...
```

The **folder name is the identity label** — filenames do not matter (they often have stale plate prefixes auto-generated by the binding pipeline; ignore those). The current production training set lives at:

```
D:\Work\Spectech\Projects\Damanat PMS Cameras\PS AI Training Dataset\detection_images\detection_images\
```

#### Pipeline

```powershell
# Run from the project root with the project venv on PATH (or call .venv\Scripts\python.exe explicitly).

# 1. T-1: Auto-crop the source dataset to tight vehicle bboxes via YOLO11s.
.venv\Scripts\python.exe tools\curate_facility_dataset.py
# Output: data/facility_top20/all/<PLATE>/*.jpg + data/facility_top20/curation_report.json
# Acceptance: total kept >= 350; every plate retains >= 3 crops.

# 2. T-2: Class-balance, temporal split, emit pairs CSV.
.venv\Scripts\python.exe tools\split_facility_dataset.py
# Output: data/facility_top20/{train,eval}/<PLATE>/*.jpg, pairs_eval.csv, split_report.json
# Acceptance: train >= 200, eval >= 100, every plate in both splits.

# 3. T-3: Fine-tune OSNet-AIN. ~25 s/epoch on CPU; 40 epochs ~= 17 minutes.
.venv\Scripts\python.exe tools\finetune_osnet_top20.py --epochs 40 --patience 8
# Output: models/osnet_facility_finetune_<YYYYMMDD>.pt and matching .log
# Acceptance: cosine margin >= 0.10 (baseline 0.054); rank-1 >= 60%.

# 4. T-4: Re-export the best checkpoint to OpenVINO INT8.
.venv\Scripts\python.exe tools\export_osnet_openvino.py `
    --checkpoint models\osnet_facility_finetune_<YYYYMMDD>.pt `
    --output-dir models\osnet_facility_int8_256x128 `
    --input-size 256x128 `
    --calibration-dir data\facility_top20\eval
# Output: models/osnet_facility_int8_256x128/model.{xml,bin,onnx,metadata.yaml}
# Acceptance: cosine drift vs torch checkpoint <= 0.020; latency <= 5 ms median.

# 5. T-5: Bench fine-tuned vs current production IR side-by-side.
.venv\Scripts\python.exe tests\test_facility_match_accuracy.py
# Output: data/facility_top20/eval_report.json + stdout summary table.
# Acceptance: fine-tuned beats baseline on >= 3 of 4 metrics (rank1, rank5, mAP, margin).

# 6. T-6: Swap the IR if T-5 passes.
# Edit config.yaml line ~215:
#   reid_openvino_model_dir: "models/osnet_facility_int8_256x128"
# Optionally re-run threshold calibration on the new IR's cosine distribution.
# Restart the engine; watch MATCH_EVENT logs for 30 minutes.
```

#### Adding a new identity

1. Create a new `<PLATE>/` folder under the source dataset root.
2. Drop in ≥ 5 captures (an ANPR entry shot plus a few floor crops gives the best multi-camera spread).
3. Append the plate to the `VERIFIED_TOP20` tuple at the top of `tools/curate_facility_dataset.py`. (The curator only walks the verified list to avoid sweeping in unlabelled images.)
4. Re-run the full T-1 → T-5 pipeline.

#### Rollback

The previous IR is preserved at `models/osnet_openvino_int8_256x128/`. To roll back: change `config.yaml` line ~215 back to `"models/osnet_openvino_int8_256x128"` and restart the engine. **No code change needed.**

#### Notes on `_Unmatched` recovery (deferred)

The 689-image `_Unmatched/` bin in the source dataset has 23 files prefixed `BAR-6998_*`, 8 prefixed `AGA-6649_*`, and similar. A scripted move-by-prefix could lift those into either existing or new plate folders, roughly doubling the long-tail folder sizes. Not implemented in the current execution; flagged as a follow-up.

### Plan reference

The full architectural rationale and design decisions for this work live in `C:\Users\ahmed\.claude\plans\alright-let-s-make-a-merry-pony.md`. Commit log from `a4a4244` onwards traces each phase:

```
Phase 0 — MatchDecision chokepoint + MatchingConfig + plugin ABCs + DI seams
Wave 1 (5 parallel workstreams):
  WS-A — OpenVINO INT8 ReID
  WS-B — color classifier
  WS-C — body-type classifier
  WS-D — PaddleOCR plate cross-check
  WS-E — fault-injection test fixtures (12 scenarios)
Phase 2 — ensemble rule + MatchVoter + threshold calibration + accuracy bench
Phase 3 — OSNet facility fine-tune + FAISS-CPU gallery index
```
