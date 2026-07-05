# Plan: Capture car snapshots from the main (high-res) stream

## Context

Today every camera is processed on the **sub-stream** (`/Streaming/Channels/102`, 720p) for CPU
efficiency, and **all car snapshots are bbox crops taken from that same low-res processing frame**.
There is no separate snapshot stream. The user wants the saved car snapshots (the images used for
ReID identity and shown in the UI) to be **high resolution**, sourced from the **main stream**
(`/Streaming/Channels/101`, 4K).

Decisions taken with the user:
- **Scope:** all cameras that already capture+save snapshots — but made **opt-in per camera**.
- **Approach:** *process the selected cameras on the main stream* (channel 101), rather than running a
  second parallel capture. This reuses all existing crop/scaling logic with **no bbox remapping and no
  frame-sync issues**.
- **Status:** plan only for now. Roll out is gated on validating the FPS impact (current effective FPS
  is already ~0.2/cam), so the feature must default **off** and be enabled per camera incrementally.

Why this works cleanly: YOLO runs at `imgsz=480` (`DetectorConfig.imgsz`, `src/config.py`) regardless
of input resolution, so switching a camera to 4K does **not** increase inference cost — it only
increases RTSP decode + per-frame preprocessing/ROI-mask cost. The crop, however, is taken from the
full 4K frame, so the snapshot is natively high-res. Slot polygons and the ROI mask already scale from
the 1280×720 reference resolution to the actual stream resolution via `build_polygon`
(`src/models/slot.py:143`) wired through `get_resolution()` in `_build_camera_pipeline`
(`src/core/engine/engine_runtime.py:508-540`), so everything adapts automatically.

## Which cameras "already take snapshots"

- **CAM-01** — `Park_Entry` zone → identity candidate snapshot (`_process_park_entry_zone`).
- **CAM-03** — `B1_Entrence` confirmation gallery: entry/deep/exit crops (`_process_confirmation_zone`
  in `src/core/engine/engine_tracking.py`). Note: the externally-pushed `B-entry` API image is a
  *separate* path whose resolution is set by the sender — this plan only affects the **internal**
  vision-capture crop for CAM-03.
- **Any camera with parking slots** — per-slot occupancy snapshots saved to
  `{snapshot_base_dir}/slot_{slot_id}_latest.jpg` (`src/core/engine/engine_runtime.py` ~87-113).

The opt-in list lets the user choose which of these to upgrade as FPS budget allows (natural first
candidates: CAM-01 and CAM-03 for identity quality).

## Implementation

### 1. Opt-in config field (YAML-only, survives DB sync)
`sync_app_config_from_db` (`src/services/config_service.py:79`) only overrides a fixed set of fields,
so a new `ProcessingConfig` field is **not** clobbered by the DB sync — no DB schema/ORM/pydantic
changes needed.

- `src/config.py` — add to `ProcessingConfig` (around line 54):
  ```python
  # Camera IDs to process on the MAIN stream (channel 101) instead of the global
  # stream_channel, so their saved car snapshots are high-resolution. Opt-in:
  # empty = legacy behaviour (all cameras on the global sub-stream).
  main_stream_cameras: List[str] = field(default_factory=list)
  ```
- `src/config.py` `load_config()` — parse `processing.main_stream_cameras` from `config.yaml`
  (list of strings) alongside the existing `processing` fields.
- `config.example.yaml` — document the new key under `processing:` with a commented example
  (`main_stream_cameras: []  # e.g. ["CAM-01", "CAM-03"]`).

### 2. Resolve per-camera channel when building RTSP URLs
- `src/core/engine/engine_runtime.py` `_build_camera_configs()` (~lines 425-440): when building each
  `CameraConfig`, choose the channel per camera:
  ```python
  main_cams = set(self.config.processing.main_stream_cameras or [])
  channel = 101 if camera.id in main_cams else self.config.processing.stream_channel
  camera_config.build_rtsp_url(channel=channel)
  ```
  No other call site needs changing — `build_rtsp_url` (`src/camera_manager.py:39`) already takes a
  channel argument.

### 3. Everything downstream auto-adapts (no code changes required)
- Stream resolution is read at open (`CameraStream.open`, `frame_width/height`).
- Slot polygons + ROI mask scale ref→actual via existing machinery (`build_polygon`,
  `_build_camera_pipeline`).
- Snapshot crops (`_crop_detection`, `_process_confirmation_zone`, `_process_park_entry_zone`, slot
  snapshots) index the full-resolution frame, so the saved images become high-res automatically.
- ReID feature extraction (`_persist_session_gallery`, `update_park_entry_candidate_snapshot`) runs on
  the higher-res crops — the intended quality win.

### 3a. Slot / boundary / ROI coordinates — NO redrawing required (key point)
The user's concern: "slots and boundaries have to change coordinates because we go 720p → 4K."
Good news: **they do not need to be redrawn or edited.** All three are stored at the fixed
**reference resolution (1280×720)** in the DB and scaled to the *actual* stream resolution at load
time — this already happens today (cameras already run at different actual sizes, e.g. CAM-EXIT is
1920×1080 on its sub-stream). The same code path handles 4K with a larger scale factor:
- **Slots + special zones + ROI** → `load_camera_slots` → `build_polygon(ref, actual)`
  (`src/services/parking_service.py:127`, `:143`).
- **Boundaries** → `load_camera_boundaries` → `build_polygon(ref, actual)`
  (`src/services/parking_service.py:172-201`).
- `actual_resolution` comes from `get_resolution(camera_id)` in `_build_camera_pipeline`
  (`engine_runtime.py:508-540`).

For a 4K main stream this is just `sx = 3840/1280 = 3.0`, `sy = 2160/720 = 3.0` — applied
automatically. **The only requirement** is that the main stream keeps the **same 16:9 aspect ratio**
as the 1280×720 reference (Hikvision main/sub normally match). If any camera's main stream has a
different aspect ratio than its reference, the polygons would stretch — so verify per camera (see
Verification step 3). No `draw_slots.py` / `roi_selector.py` re-run is needed otherwise.

### 4. Where snapshots are saved (for validation before full rollout)
Snapshots continue to flow through the existing storage so the change is transparent:
- Identity images → `vehicle_images/` (`_write_snapshot_file`, `_store_session_reference_snapshots`
  in `src/vehicle_registry/vehicle_registry_core.py`).
- Slot images → `{snapshot_base_dir}/slot_{slot_id}_latest.jpg`.

To make validation easy without trusting the change yet, add a lightweight, opt-in eval dump: when a
camera is in `main_stream_cameras`, also write a copy of each saved identity crop to
`vehicle_images/main_stream_eval/{camera_id}/{plate-or-token}_{timestamp}.jpg`. This gives a side
folder to eyeball resolution/quality and confirm the gain before enabling broadly. (Implement as a
small helper next to `_write_snapshot_file`; guard behind the same opt-in list.)

## Performance & phased rollout (the FPS gate)

- Extra cost per main-stream camera = 4K RTSP decode + preprocessing/ROI-mask on an ~8× larger array
  per processed frame. **YOLO inference cost is unchanged** (imgsz=480).
- Because the system is already at ~0.2 FPS/cam, **enable one camera at a time** and watch the
  `[INFO] effective FPS/camera` line:
  1. Start with **CAM-03** only → measure FPS over a few windows.
  2. Add **CAM-01** → measure again.
  3. Expand to slot-snapshot cameras only if headroom remains.
- If decode cost is too high but high-res snapshots are still wanted, the fallback is the *separate
  main-stream grab* approach (keep processing on sub, open a second main capture, up-scale the bbox) —
  out of scope here, noted only as the escape hatch.

## Files to modify
- `src/config.py` — new `ProcessingConfig.main_stream_cameras` field + YAML parsing in `load_config()`.
- `config.example.yaml` — document the new key.
- `src/core/engine/engine_runtime.py` — per-camera channel selection in `_build_camera_configs()`.
- `src/vehicle_registry/vehicle_registry_core.py` — optional eval-dump helper for validation copies
  (guarded by the opt-in list).

## Verification
1. **Unit/smoke:** set `processing.main_stream_cameras: ["CAM-03"]` in `config.yaml`; start with
   `python main.py --api --show`. Confirm the startup line shows CAM-03 connected at the **4K** main
   resolution (e.g. `[3840×2160]` or the camera's main-stream size) while other cameras still show
   720p.
2. **Snapshot quality:** drive a car through CAM-03's `B1_Entrence` zone; confirm the saved
   `session_*.jpg` (and the `main_stream_eval/CAM-03/` copy) are high-resolution vs. the previous 720p
   crops.
3. **Scaling sanity (the coordinate concern):** in `--show`, confirm slot polygons, special zones,
   boundary bands, and the ROI mask still line up on the 4K frame — they should, because they auto-scale
   ×3 from the 1280×720 reference. If any camera's main stream is NOT 16:9, the overlays will look
   stretched on that camera → that camera needs its main-stream resolution checked (or, worst case, a
   re-draw). Also confirm occupancy + boundary crossings still trigger normally.
4. **FPS impact:** watch `[INFO] effective FPS/camera` for several windows with the camera enabled vs.
   disabled; record the delta. This is the data the user needs before expanding the opt-in list.
5. **Regression:** with `main_stream_cameras: []` (default), behaviour is byte-for-byte the legacy
   sub-stream path — confirm nothing changes.
