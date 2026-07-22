# Damanat PMS Video Analytics — System Functionality Document

> **Version:** 1.0 | **Last Updated:** March 19, 2026

A CPU-optimized, real-time parking management system that processes 14 RTSP camera feeds to monitor parking slot occupancy, detect vehicle violations, identify license plates via ANPR integration, and provide a REST API for external systems.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Camera Infrastructure](#2-camera-infrastructure)
3. [Core Processing Pipeline](#3-core-processing-pipeline)
4. [Parking Slot Management](#4-parking-slot-management)
5. [ANPR Integration & Plate Assignment](#5-anpr-integration--plate-assignment)
6. [Vehicle Image Matching](#6-vehicle-image-matching)
7. [Violation Detection System](#7-violation-detection-system)
8. [REST API Endpoints](#8-rest-api-endpoints)
9. [Visualization & Display](#9-visualization--display)
10. [Configuration Reference](#10-configuration-reference)
11. [Running the System](#11-running-the-system)

---

## 1. System Architecture

```
┌─────────────┐     ┌─────────────────────────────────────────────────────┐
│  14 RTSP     │     │              ParkingEngine                         │
│  Camera      │────▶│                                                     │
│  Streams     │     │  CameraManager (threaded grabbers)                  │
└─────────────┘     │       │                                              │
                    │       ▼                                              │
                    │  TrackedDetector (YOLOv11 + ByteTrack)               │
                    │       │                                              │
                    │       ▼                                              │
                    │  SlotAssigner (point-in-polygon + overlap)           │
                    │       │                                              │
                    │       ▼                                              │
                    │  SlotStateMachine (per-slot, debounced)              │
                    │       │                                              │
                    │       ▼                                              │
                    │  EventBus (JSON events → stdout + logfile)           │
                    │       │                                              │
                    │  VehicleRegistry ◀── ANPR API (FastAPI)             │
                    │  ImageMatcher                                        │
                    └─────────────────────────────────────────────────────┘
```

**Key design decisions:**
- **Single YOLO model** shared across all cameras (memory-efficient)
- **Round-robin** frame processing — each camera gets ~1 FPS
- **Threaded frame grabbing** — background threads per camera prevent RTSP buffer overflow
- **Thread-safe** vehicle registry allows concurrent API + engine access

---

## 2. Camera Infrastructure

### Camera Layout

| Camera | Floor        | Role            | IP          |
|--------|-------------|-----------------|-------------|
| CAM_01 | Ground Floor| Gate (Entry)    | 10.1.13.60  |
| CAM_02 | Ground Floor| Gate (Front)    | 10.1.13.61  |
| CAM_03 | B1          | Parking         | 10.1.13.62  |
| CAM_04 | B1          | Parking         | 10.1.13.63  |
| CAM_05 | B1          | Parking         | 10.1.13.64  |
| CAM_06 | B1          | Parking         | 10.1.13.65  |
| CAM_07 | B1          | Parking         | 10.1.13.66  |
| CAM_08 | B1          | Parking         | 10.1.13.67  |
| CAM_09 | B2          | Parking         | 10.1.13.68  |
| CAM_10 | B2          | Parking         | 10.1.13.69  |
| CAM_11 | B2          | Parking         | 10.1.13.70  |
| CAM_12 | B2          | Parking         | 10.1.13.71  |
| CAM_13 | B2          | Parking         | 10.1.13.72  |
| CAM_14 | B2          | Parking         | 10.1.13.73  |

### Stream Management

- **Protocol:** RTSP over TCP
- **Resolution:** Sub-stream 720p (channel 102) for CPU efficiency; main stream 4K (channel 101) available
- **Frame Grabbing:** Per-camera background daemon threads continuously read frames, storing only the latest
- **Reconnection:** Automatic reconnection every 10s on stream failure
- **Buffer:** `CAP_PROP_BUFFERSIZE = 1` to minimize latency

### Special Camera Roles

- **CAM_01 (Gate Camera):** When the ANPR server sends a plate without an image, the system automatically captures vehicle crops from CAM_01's latest frame. These crops are stored as reference images for visual matching.

---

## 3. Core Processing Pipeline

Each frame goes through the following stages:

### 3.1 Detection & Tracking

- **Model:** YOLOv11n (OpenVINO optimized for CPU)
- **Classes:** COCO class `2` (cars only)
- **Confidence threshold:** 0.35
- **Input resolution:** 640px
- **Tracker:** ByteTrack (persistent IDs across frames via `model.track(persist=True)`)

Output: list of `Detection` objects with `bbox`, `class_id`, `confidence`, `track_id`.

### 3.2 Gate Camera Snapshot

For CAM_01 specifically, the engine feeds the latest frame + detections to `VehicleRegistry.update_gate_snapshot()` on every frame. This data is used later for automatic crop capture when ANPR events arrive without images.

### 3.3 Plate Assignment (B1 + B2 floors only)

For every detected car without a plate on B1/B2 floors:
1. Crop the car from the frame
2. Call `VehicleRegistry.try_match_by_image()` to compare against stored ANPR/gate reference images
3. If score ≥ threshold → assign the plate to that track ID

### 3.4 Slot Assignment

Uses a two-tier strategy:

1. **Primary (point-in-polygon):** Compute the bottom-center of each vehicle's bounding box. Test against each slot polygon using Shapely.
2. **Fallback (overlap ratio):** If bottom-center misses, compute intersection area between bbox and slot polygon. Assign if overlap > 30%.
3. **Tie-breaking:** If multiple vehicles compete for one slot, the closest to the slot centroid wins.

### 3.5 State Machine (per slot)

Each parking slot has an independent 4-state finite state machine with debounce:

```
VACANT ──[vehicle detected]──▶ ENTERING
  ▲                                │
  │ [vehicle disappears          [stays for 5 frames]
  │  before 5 frames]              │
  │                                ▼
  ◀──[absent 8 frames]──── OCCUPIED
         LEAVING ◀──[vehicle gone]──┘
              │
              │ [re-detected]
              ▼
           OCCUPIED
```

| Transition | Frames Required | Event Emitted |
|---|---|---|
| VACANT → ENTERING | 1 | `vehicle_entering` |
| ENTERING → OCCUPIED | 5 (confirm_enter) | `vehicle_parked` |
| OCCUPIED → LEAVING | 1 | `vehicle_leaving` |
| LEAVING → VACANT | 8 (confirm_leave) | `slot_vacant` |
| LEAVING → OCCUPIED | 1 (re-detected) | *(none)* |
| ENTERING → VACANT | 1 (disappeared) | *(none)* |

### 3.6 Event Emission

Events are emitted as JSON to stdout and optionally to a log file:

```json
{
  "event": "vehicle_parked",
  "slot_id": "B2",
  "track_id": -100,
  "timestamp": "2026-03-16T17:11:00.697570",
  "camera_id": "CAM_04",
  "floor": "B1"
}
```

Event types: `vehicle_entering`, `vehicle_parked`, `vehicle_leaving`, `slot_vacant`, `vehicle_violation`.

---

## 4. Parking Slot Management

### Slot Definition

Slots are defined per camera in JSON files (`slots/b1_cam04.json`):

```json
[
  {
    "id": "B2",
    "polygon": [[305, 214], [476, 203], [525, 480], [275, 480]],
    "label": "Slot B2"
  }
]
```

- Polygons are pixel coordinates matching the camera view
- Each camera has its own slots file
- Slots can be drawn interactively using `draw_slots.py`
- Slot IDs containing "violation" are treated as **violation zones**

### Slot Tool (`draw_slots.py`)

An interactive tool for defining slot polygons:
- Opens a camera stream snapshot
- Click to define polygon corners
- Saves to JSON for engine use

---

## 5. ANPR Integration & Plate Assignment

### How ANPR Works

1. External ANPR server detects a license plate at the parking gate
2. Sends a `POST` request to this server with plate + optional image
3. System registers the plate in a pending queue
4. System attempts to match the plate to a detected car via:
   - **Image Matching** (visual comparison using gate crops)
   - **Queue Assignment** (FIFO — oldest plate → first unplated car)

### Gate Camera Crop Capture

When an ANPR event arrives **without** an image:
1. System captures all detected car crops from CAM_01's latest frame
2. Stores them as reference images in `vehicle_images/`
3. These crops serve as the reference for visual matching against parking cameras

### Vehicle Record

Each ANPR event creates a `VehicleRecord`:

| Field | Description |
|---|---|
| `plate` | License plate string |
| `direction` | `"entry"` or `"exit"` |
| `timestamp` | When the event was received |
| `image_path` | Saved image on disk |
| `anpr_image` | In-memory image(s) for matching |
| `linked_slot` | Slot ID once parked |
| `linked_camera` | Which camera confirmed parking |
| `linked_floor` | Floor (B1, B2) |
| `track_id` | ByteTrack ID |

### Data Flow

```
ANPR Server ──POST──▶ /api/anpr/event
                          │
                  register_anpr_event()
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
     Image provided?            No image?
     Save + decode             _capture_gate_crops()
              │                       │
              └───────┬───────────────┘
                      ▼
              _pending_entries queue
                      │
              ┌───────┴───────┐
              ▼               ▼
       try_match_by_image  try_assign_plate
       (visual matching)   (FIFO queue)
              │               │
              └───────┬───────┘
                      ▼
           _track_plate_map[(cam, track)] = plate
           _parked[slot_id] = record
```

---

## 6. Vehicle Image Matching

### Multi-Feature Matcher (`VehicleImageMatcher`)

Compares two vehicle images using 5 independent features:

| Feature | Weight | Method |
|---|---|---|
| **Dominant Color** | 35% | K-means clustering (3 clusters) in LAB color space |
| **Color Histogram** | 25% | LAB channel histograms with correlation comparison |
| **Regional Color** | 20% | Top/bottom half mean color comparison (roof vs body) |
| **SSIM** | 10% | Structural Similarity Index (luminance, contrast, structure) |
| **Edge Density** | 10% | Canny edge detection, 4-quadrant density comparison |

### Key Design Choices

- **LAB color space** used everywhere (perceptually uniform — distances match human perception)
- All images resized to small fixed sizes (64×64 or 128×128) for speed
- Matching only runs on **B1 and B2 floor cameras** (not gate/ground floor)
- Threshold: **35%** similarity required for a match
- Lazy-loaded matcher (first use initializes it)

### Debug Output

When enabled, the system logs per-feature score breakdowns:

```
[MATCH-DEBUG] Plate=ZUR-8870 crop#0 vs Track:-100 (cam=CAM_04) → score=0.412
    [FEATURES] dominant_color=0.65 | color_hist=0.43 | regional_color=0.28 | ssim=0.15 | edge=0.72 → final=0.412
```

---

## 7. Violation Detection System

### How It Works

Slots with IDs containing `"violation"` are treated as no-parking zones. When a car is confirmed parked in such a zone on the Ground Floor:

1. The car is cropped from the frame
2. Compared against recent violators using the image matcher (threshold: 40%)
3. If it's a **new** violator (not a duplicate):
   - Event type is changed to `"vehicle_violation"`
   - Crop is stored in recent violator history (30s window)
   - Alert is printed: `[ALERT] New Violation!`
4. If it **matches** a recent violator → treated as duplicate (no re-alert)

### Visual Feedback

- Violation zones with occupied vehicles show **flickering red** polygons
- Label displays `"!!! VIOLATION !!!"` instead of the slot ID
- Thicker border (4px vs normal 2px)

---

## 8. REST API Endpoints

Base URL: `http://0.0.0.0:8000` (started with `--api` flag)

### ANPR

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/anpr/event` | Legacy JSON ANPR; V2 modes require service auth, authoritative allows exits only |
| `POST` | `/api/anpr/event/upload` | Legacy multipart ANPR; HTTP 410 in V2 modes |
| `POST` | `/api/line-crossing` | Legacy entry image; HTTP 410 in V2 modes |

**JSON body for `/api/anpr/event`:**
```json
{
  "plate": "ZUR-8870",
  "direction": "entry",
  "image_base64": "<base64 JPEG>",
  "camera_id": "ANPR_01",
  "timestamp": "2026-03-16T17:00:00"
}
```

### Slot Status

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/slots` | All slot statuses across all cameras |
| `GET` | `/api/slots/{floor}` | Slot statuses for a specific floor (e.g., `B1`) |

**Response includes:** `slot_id`, `state`, `occupied`, `assigned_track_id`, `camera_id`, `floor`, `plate`

### Vehicle Queries

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/vehicle/{plate}` | Find where a specific plate is parked |
| `GET` | `/api/vehicles` | All currently parked vehicles |
| `GET` | `/api/vehicles/pending` | Vehicles that entered but haven't been assigned a slot |

### System

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/stats` | Summary statistics (total/occupied/vacant slots, parked/pending counts) |
| `GET` | `/api/health` | Health check |

### Interactive Docs

Swagger UI available at: `http://localhost:8000/docs`

---

## 9. Visualization & Display

### Multi-Camera Grid View

When `--show` is used, cameras are displayed in **per-floor grid windows**:
- 3 columns per floor
- Each cell is 480×270 pixels
- Window title: `Damanat PMS — B1`, `Damanat PMS — Ground Floor`, etc.

### Annotations

Each frame is annotated with:
- **Slot polygons** colored by state (Green=VACANT, Yellow=ENTERING, Red=OCCUPIED, Orange=LEAVING)
- **Slot labels** showing `{slot_id}: {state}`
- **Detection bounding boxes** (when in single-camera mode)
- **Camera ID and floor** as header text
- **Violation alerts** with flickering red polygons and `!!! VIOLATION !!!` labels
- **Plate numbers** displayed on assigned vehicles

### Single-Camera Debug View

Using `--show-camera CAM_04` opens only that camera's view.

---

## 10. Configuration Reference

All settings are in `config.yaml`:

### Processing

| Key | Default | Description |
|---|---|---|
| `mode` | `round_robin` | Camera cycling strategy |
| `target_fps_per_camera` | `1` | Frames per second per camera |
| `stream_channel` | `102` | RTSP channel (101=4K, 102=720p) |

### Detector

| Key | Default | Description |
|---|---|---|
| `model_path` | `models/yolo11n_openvino_model` | YOLO model path |
| `confidence` | `0.35` | Minimum detection confidence |
| `classes` | `[2]` | COCO classes to detect (2=car) |
| `imgsz` | `640` | Inference input resolution |

### Motion Scheduler (single-process mode)

`shadow` and `enforce` require `VA_SINGLE_PROCESS=1`; unsupported runtime paths
fail closed at startup. In `enforce`, `analysis_fps` must be at least
`processing.target_fps_per_camera` for every non-bypass camera; entry cameras
and entrance-zone cameras remain bypassed.

The deployment variables `VA_MOTION_SCHEDULER_MODE` and `VA_SLOT_STATE_MODE`
override the YAML `motion_scheduler.mode` and `state_machine.mode`
respectively. Accepted values are `legacy|shadow|enforce` for motion and
`legacy|shadow|time` for slot state.

| Key | Default | Description |
|---|---|---|
| `mode` | `legacy` | `legacy` disables motion analysis; `shadow` measures; `enforce` gates quiet frames |
| `analysis_fps` | `2.0` | Per-camera frame-difference rate; in `enforce`, must be ≥ target FPS unless bypassed |
| `analysis_width` | `96` | Frame-difference width in pixels |
| `pixel_delta` | `18` | Per-pixel grayscale change threshold |
| `changed_ratio` | `0.02` | Changed-pixel ratio that marks motion active |
| `active_hold_seconds` | `2.0` | Keep motion active after the latest changed frame |
| `sentinel_interval_seconds` | `5.0` | Target quiet interval for mandatory YOLO checks |
| `stale_frame_seconds` | `3.0` | Frame age after which inference output is treated as UNKNOWN |
| `always_infer` | `false` | Bypass motion gating globally |
| `camera_overrides` | `{}` | Per-camera overrides for the fields above except `mode` |

### State Machine

| Key | Default | Description |
|---|---|---|
| `confirm_enter_frames` | `5` | Frames to confirm OCCUPIED |
| `confirm_leave_frames` | `8` | Frames to confirm VACANT |
| `mode` | `legacy` | `legacy` frame counters, `shadow` time-policy diagnostics, or authoritative `time` mode |
| `enter_seconds` | `3.0` | Minimum PRESENT duration in `shadow`/`time` |
| `leave_seconds` | `20.0` | Minimum ABSENT duration in `shadow`/`time` |
| `enter_min_observations` | `2` | Minimum PRESENT observations in `shadow`/`time` |
| `leave_min_observations` | `3` | Minimum ABSENT observations in `shadow`/`time` |
| `enter_cancel_seconds` | `1.0` | Sustained ABSENT time required to cancel ENTERING |
| `enter_cancel_min_observations` | `2` | ABSENT observations required to cancel ENTERING |
| `leave_start_seconds` | `1.0` | Sustained ABSENT time before OCCUPIED enters LEAVING |
| `leave_start_min_observations` | `2` | ABSENT observations before OCCUPIED enters LEAVING |
| `max_known_gap_seconds` | `8.0` | Restart a timed evidence run after a longer gap |

### Slot Assigner

| Key | Default | Description |
|---|---|---|
| `overlap_threshold` | `0.3` | Minimum bbox-polygon overlap for assignment |

---

## 11. Running the System

### Prerequisites

```bash
pip install ultralytics opencv-python shapely pyyaml fastapi uvicorn
```

### Run Modes

```bash
# Multi-camera with API (production)
python main.py --api

# Multi-camera with visualization
python main.py --show --api

# Single camera debug
python main.py --camera CAM_04 --show

# Custom port
python main.py --api --port 9000

# Legacy single video
python main.py --video sample.mp4 --show
```

### Key Files

| File | Purpose |
|---|---|
| `main.py` | Entry point, CLI argument parsing |
| `src/core/engine.py` | Main processing loop (multi + single camera) |
| `src/core/slot_assigner.py` | Vehicle → slot polygon assignment |
| `src/detection/detector.py` | YOLO inference wrapper |
| `src/detection/tracker.py` | ByteTrack tracking wrapper |
| `src/models/state_machine.py` | Per-slot FSM with debounce |
| `src/models/slot.py` | Parking slot data model + JSON loader |
| `src/events/event_bus.py` | Structured JSON event emission |
| `src/camera_manager.py` | Multi-stream RTSP manager |
| `src/vehicle_registry.py` | Plate → slot tracking, ANPR registration |
| `src/image_matcher.py` | Multi-feature vehicle visual matching |
| `src/api.py` | FastAPI REST server |
| `src/config.py` | YAML config loader |
| `draw_slots.py` | Interactive slot polygon tool |
| `config.yaml` | Production camera + system config |
