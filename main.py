"""
main.py — Entry point for the Damanat PMS Video Analytics system.

Supports three modes:
  1. Multi-camera mode (default): processes all cameras from config.yaml
  2. Single-camera mode: --camera or --video flag
  3. API server mode: --api flag starts FastAPI alongside the engine

Usage:
    python main.py                                     # Multi-camera mode
    python main.py --api                               # Multi-camera + API server
    python main.py --camera CAM_04 --show              # Single camera with visualization
    python main.py --video sample.mp4 --show           # Legacy single-file mode
    python main.py --show --show-camera CAM_04         # Multi-camera, visualize one

Press 'q' in the visualization window to quit (if --show is used).
Press Ctrl+C to stop at any time.
"""

import argparse
import os
import sys
import threading

# Force UTF-8 console output. Windows consoles default to a legacy code page
# (e.g. cp1252), which renders UTF-8 names like "B1 Parking — Camera 08" as
# mojibake ("B1 Parking â€” Camera 08"). The names are stored correctly; only
# the printout was garbled. reconfigure() is a no-op where stdout isn't a
# regular stream (e.g. already-wrapped pipes), so guard it defensively.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# Pin to the CPU slice the supervisor handed us — MUST happen before cv2/torch/
# openvino are imported, because TBB and OpenCV size their thread pools from the
# affinity mask exactly once, at import/first-use. See src/cpu_affinity.py.
from src.cpu_affinity import apply_from_env as _apply_cpu_affinity

_PINNED_CPUS = _apply_cpu_affinity()
if _PINNED_CPUS:
    print(f"[INFO] pinned to {len(_PINNED_CPUS)} CPUs: {_PINNED_CPUS}")

# Unpinned (the supervisor found a CFS quota, so the affinity mask is the whole
# node and pinning would nail us to host cores we don't own): OpenCV would size
# its pool to every core on the node, in every group at once. Cap it explicitly —
# the pool cap is what the pinning was really buying. OpenVINO's equivalent cap
# rides in via VA_OV_NUM_THREADS -> detector.ov_num_threads (see src/config.py).
_CV_THREADS = os.environ.get("VA_CV_NUM_THREADS", "").strip()
if _CV_THREADS:
    try:
        import cv2

        cv2.setNumThreads(max(1, int(_CV_THREADS)))
        print(f"[INFO] unpinned: OpenCV thread pool capped at {_CV_THREADS}")
    except (ImportError, ValueError) as _exc:
        print(f"[WARN] could not cap OpenCV threads to {_CV_THREADS!r}: {_exc!r}")

from src.logging_setup import configure_logging
from src.config import load_config
from src.core.engine import ParkingEngine
from src.database import init_db
from src.services.config_service import (
    ensure_areas_initialized,
    ensure_config_initialized,
    load_cameras_from_db,
    sync_app_config_from_db,
    sync_areas_from_db,
)

def start_api_server(
    engine,
    registry,
    host="0.0.0.0",
    port=8000,
    entry_coordinator=None,
):
    """Start the FastAPI server in a background thread."""
    import uvicorn
    from src.api import create_app

    def get_slot_statuses():
        """Callback for the API to get current slot statuses."""
        all_statuses = []
        for cam_id, pipeline in engine.pipelines.items():
            slot_meta = {s.id: s for s in pipeline.slots}
            for sm in pipeline.state_machines.values():
                status = sm.get_status()
                status["camera_id"] = cam_id
                status["floor"] = pipeline.floor
                slot = slot_meta.get(sm.slot_id)
                status["label"] = slot.label if slot else sm.slot_id
                status["slot_name"] = slot.label if slot else sm.slot_id
                status["zone_id"] = slot.zone_id if slot else ""
                status["zone_name"] = slot.zone_name if slot else status["label"]
                all_statuses.append(status)
        return all_statuses

    def get_camera_frame(cam_id: str):
        if hasattr(engine, "cam_manager") and engine.cam_manager:
            return engine.cam_manager.read_camera(cam_id)
        return False, None

    def get_slot_snapshot_source(slot_id: str):
        for cam_id, pipeline in engine.pipelines.items():
            for slot in pipeline.slots:
                if slot.id == slot_id:
                    return {
                        "camera_id": cam_id,
                        "slot": slot,
                        "state_machine": pipeline.state_machines.get(slot_id),
                        "floor": pipeline.floor,
                    }
        return None

    def get_park_entry_crop(cam_id: str):
        snapshot_cam_id = "CAM_03"
        crop = engine.get_recent_zone_vehicle_crop(
            snapshot_cam_id,
            zone_name_hint="Entrence",
        )
        if crop is not None and crop.size > 0:
            return True, crop
        return False, None

    def detect_vehicle_crop(frame):
        """Detect vehicles in a one-off image (e.g. a CAM-03 line-crossing
        frame) and return the largest vehicle crop — the entering car, which
        dominates the frame over background/parked cars. Returns None if no
        vehicle is found. Uses a plain predict (no tracker state mutation)."""
        try:
            det = engine.detector
            results = det.model.predict(
                frame,
                conf=det.detector_config.confidence,
                classes=det.detector_config.classes,
                imgsz=det.detector_config.imgsz,
                device=det.device,
                verbose=False,
            )
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                return None
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            x1, y1, x2, y2 = (int(v) for v in xyxy[int(areas.argmax())])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            return frame[y1:y2, x1:x2].copy()
        except Exception as exc:
            print(f"[API] detect_vehicle_crop failed: {exc!r}")
            return None

    app = create_app(
        vehicle_registry=registry,
        get_slot_statuses=get_slot_statuses,
        get_camera_frame=get_camera_frame,
        get_slot_snapshot_source=get_slot_snapshot_source,
        get_park_entry_crop=get_park_entry_crop,
        detect_vehicle_crop=detect_vehicle_crop,
        get_engine_status=engine.get_engine_status,
        event_bus=engine.event_bus,
        db_manager=engine.db_manager,
        snapshot_base_dir=engine.config.output.snapshot_base_dir,
        public_base_url=engine.config.output.public_base_url,
        snapshot_url_prefix=engine.config.output.snapshot_url_prefix,
        gateway_path_prefix=engine.config.output.gateway_path_prefix,
        entry_coordinator=entry_coordinator,
    )

    # Include routers
    from src.routers.parking_router import router as parking_router
    from src.routers.slot_status_router import router as slot_status_router
    from src.routers.alert_router import router as alert_router
    app.include_router(parking_router)
    app.include_router(slot_status_router)
    app.include_router(alert_router)

    def run_server():
        uvicorn.run(app, host=host, port=port, log_level="info")

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    print(f"[INFO] API server started at http://{host}:{port}")
    print(f"[INFO] Docs: http://{host}:{port}/docs\n")
    return thread


def main():
    # FIRST statement, before argparse and before any subsystem logs. Until this
    # runs the root logger has no handler and every logger.info() in src/ is
    # discarded by logging.lastResort (level WARNING). See src/logging_setup.py.
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Damanat PMS Video Analytics — CPU-optimized parking slot detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                  # All cameras, round-robin
  python main.py --api                            # Multi-camera + API server
  python main.py --api --port 9000                # Custom API port
  python main.py --camera CAM_04 --show           # Single camera with visualization
  python main.py --video sample.mp4 --show        # Legacy single-file mode
  python main.py --show --show-camera CAM_04      # Multi-cam, show one camera
  python main.py --reset-plates-only              # Wipe all slot plates, then exit
  python main.py --reset-plates --api             # Clear plates, then run clean
        """,
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML config file (default: config.yaml).",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Single video source (file or RTSP URL) — legacy mode.",
    )
    parser.add_argument(
        "--camera", type=str, default=None,
        help="Run only a specific camera by ID (e.g., CAM_04).",
    )
    parser.add_argument(
        "--floor", type=str, default=None,
        help="Run only the cameras on a given floor (e.g., B1) — for testing "
             "one floor in isolation. Applies to multi-camera mode.",
    )
    parser.add_argument(
        "--cameras", type=str, default=None,
        help="Run only a comma-separated subset of cameras by ID (e.g. "
             "'CAM-03,CAM-04,CAM-05'). Splits a floor across parallel processes "
             "to raise aggregate FPS (each process is one serial inference "
             "worker; keep <=6 cameras per process to approach 5 fps/cam). "
             "Case-insensitive; '_' and '-' are interchangeable. Composes with "
             "--floor. Applies to multi-camera mode.",
    )
    parser.add_argument(
        "--reset-plates", action="store_true",
        help="Wipe every slot's plate identity (current_plate / lock + latest "
             "slot_status plate) at startup, then run with a clean slate. "
             "Occupancy is left untouched; nothing is restored on next boot. "
             "VA-local only (does not notify PMS-AI).",
    )
    parser.add_argument(
        "--reset-plates-only", action="store_true",
        help="Same clear as --reset-plates, then exit without running the engine.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show annotated video window for debugging.",
    )
    parser.add_argument(
        "--show-camera", type=str, default=None,
        help="Which camera to visualize in multi-camera mode (e.g., CAM_04).",
    )
    parser.add_argument(
        "--fps", type=int, default=None,
        help="Override target processing FPS.",
    )
    parser.add_argument(
        "--api", action="store_true",
        help="Start the FastAPI server alongside the engine.",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="API server port (default: 8000).",
    )
    parser.add_argument(
        "--supervise", action="store_true",
        help="Launch + supervise ALL per-group worker processes from this one "
             "entry point (Python equivalent of run_all.ps1). Spawns one "
             "'main.py --cameras <group>' process per group in supervisor.py's "
             "GROUPS table; exactly one group owns --api. This flag does NOT run "
             "an engine itself — it only orchestrates the workers.",
    )
    parser.add_argument(
        "--foreground", action="store_true",
        help="With --supervise: mirror every child group's log to stdout "
             "(use as the Docker PID 1). Ignored without --supervise.",
    )
    args = parser.parse_args()

    # Supervisor mode short-circuit: fan out to the per-group worker processes
    # and block, WITHOUT building an engine in this parent. Placed above
    # load_config/init_db so the orchestrating parent skips config load, DB init,
    # pipeline build and camera opening — the real per-worker startup happens in
    # the spawned 'main.py --cameras <group>' children. (For the very lightest
    # parent, run `python supervisor.py` directly, which avoids main.py's
    # module-level torch import entirely.) --reset-plates is honoured as the
    # one-shot global wipe before any worker boots.
    if args.supervise:
        import supervisor
        sys.exit(supervisor.run(
            reset_plates=args.reset_plates,
            foreground=args.foreground,
            port=args.port,
        ))

    # --- Load configuration ---
    print("=" * 60)
    print("  Damanat PMS Video Analytics")
    print("  CPU-Optimized Parking Management System")
    print("=" * 60)

    config = load_config(args.config)
    db = init_db(config.database.url)
    db.create_tables()

    # Plate reset: wipe every slot's persisted plate identity so the parking
    # starts with no plate numbers. Runs before pipelines build, so the fresh
    # engine also starts with empty in-memory state and _load_camera_db_state
    # finds nothing to restore. --reset-plates-only clears and exits.
    if args.reset_plates or args.reset_plates_only:
        from src.services.slot_status_service import reset_all_slot_plates
        from src.vehicle_registry.gallery_store import VehicleGalleryStore

        session = db.SessionLocal()
        try:
            cleared = reset_all_slot_plates(session)
            print(f"[RESET] Cleared plate identity on {cleared} slot(s).")
        finally:
            session.close()
        # Also wipe the persisted per-car galleries — otherwise a returning car
        # would warm-start its just-cleared plate identity back from disk.
        wiped = VehicleGalleryStore.clear_all(config.output.snapshot_base_dir)
        print(f"[RESET] Removed {wiped} per-car gallery folder(s).")
        if args.reset_plates_only:
            sys.exit(0)

    # Initialize Config table if empty
    session = db.SessionLocal()
    try:
        ensure_config_initialized(session, config)
        # Link DB config to runtime app_config
        sync_app_config_from_db(session, config)
        # Zoning: seed parking_areas from YAML when empty, then make the DB the
        # runtime source of truth for areas (mirrors the Config flow above).
        ensure_areas_initialized(session, config)
        sync_areas_from_db(session, config)
        # DB-first camera roster (enabled rows from the cameras table); falls
        # back to YAML cameras when the table is absent/empty. Runs after areas
        # so a camera's `area` resolves against the synced parking_areas.
        load_cameras_from_db(session, config)
    finally:
        session.close()

    # Restrict the roster to a single floor when --floor is given. Useful for
    # bringing up / load-testing one floor (e.g. B1) in isolation without the
    # full 25-camera deployment. Match is case-insensitive on the floor label.
    if args.floor:
        want = args.floor.strip().lower()
        filtered = [c for c in config.cameras if (c.floor or "").strip().lower() == want]
        if not filtered:
            floors = sorted({(c.floor or "").strip() for c in config.cameras})
            print(f"\n[ERROR] No cameras on floor '{args.floor}'.")
            print(f"[HINT] Available floors: {floors}")
            sys.exit(1)
        config.cameras = filtered
        print(
            f"[INFO] --floor {args.floor}: running {len(filtered)} camera(s): "
            f"{[c.id for c in filtered]}"
        )

    # Restrict the roster to an explicit camera subset when --cameras is given.
    # This is the parallel-scaling lever: each process runs one serial inference
    # worker (~fixed inferences/sec), so fps/camera scales as that budget split
    # over fewer cameras. Launch several processes (each with its own --cameras
    # group and --port) to use more cores concurrently. Composes with --floor.
    if args.cameras:
        def _norm(cid: str) -> str:
            return cid.strip().upper().replace("_", "-")

        wanted = [_norm(c) for c in args.cameras.split(",") if c.strip()]
        by_id = {_norm(c.id): c for c in config.cameras}
        unknown = [c for c in wanted if c not in by_id]
        if unknown:
            print(f"\n[ERROR] Unknown camera id(s) for --cameras: {unknown}")
            print(f"[HINT] Available: {sorted(by_id)}")
            sys.exit(1)
        # Preserve request order, drop duplicates.
        seen = set()
        filtered = []
        for c in wanted:
            if c not in seen:
                seen.add(c)
                filtered.append(by_id[c])
        config.cameras = filtered
        print(
            f"[INFO] --cameras: running {len(filtered)} camera(s): "
            f"{[c.id for c in filtered]}"
        )

    # Apply CLI overrides
    if args.show:
        config.output.show_video = True
    if args.show_camera:
        config.output.show_camera = args.show_camera

    # Motion scheduling is implemented only by the multi-camera,
    # single-process async loop. Validate this before loading either AI model so
    # a deployment cannot appear healthy while silently running legacy YOLO
    # scheduling.
    _single_process = os.environ.get("VA_SINGLE_PROCESS", "").strip().lower() in (
        "1", "true", "yes", "on"
    )
    from src.core.motion_scheduler import validate_motion_runtime

    try:
        validate_motion_runtime(
            config.motion_scheduler.mode,
            scheduler_available=(
                _single_process and not bool(args.video) and not bool(args.camera)
            ),
        )
    except ValueError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(2)

    # Validate model exists
    if not os.path.exists(config.detector.model_path):
        print(f"\n[ERROR] Model file not found: '{config.detector.model_path}'")
        print(f"[HINT] Run 'python setup_model.py' first to download the model.")
        sys.exit(1)

    # --- Prime the ReID matcher singleton with the runtime config ---
    # The matcher is a process-wide singleton. If it is first constructed by a
    # camera tracker / the registry (which call get_reid_matcher() with no
    # args) it silently uses MatchingConfig() defaults and ignores config.yaml
    # — falling back to the slow torchreid ImageNet path. Build it here, once,
    # with the loaded matching config so the OpenVINO IR (e.g. PS_carMatching)
    # is actually used.
    from src.reid_matcher import get_reid_matcher
    matcher = get_reid_matcher(config.preprocessing.reid, config.matching)
    print(
        f"[INFO] ReID matcher backend: {matcher.backend} "
        f"(model_dir='{config.matching.reid_openvino_model_dir}')"
    )

    # --- Initialize vehicle registry (shared between engine and API) ---
    from src.vehicle_registry import VehicleRegistry
    from src.zoning import AreaRegistry

    # Zoning: a read-only camera↔area index built from config. When no areas
    # are defined it reports enabled=False and the registry stays in legacy
    # un-zoned mode (area-bounded ReID + per-area ownership are no-ops).
    area_registry = AreaRegistry(config)
    registry = VehicleRegistry(
        image_dir=config.output.snapshot_base_dir,
        # Without this the registry silently falls back to MatchingConfig()
        # DEFAULTS — ignoring every YAML-tuned threshold AND leaving the
        # opt-in gallery persistence disabled, so no per-plate gallery folder
        # is ever seeded. Pass the loaded matching config so
        # the tuned Youden thresholds + gallery persistence actually apply.
        matching_config=config.matching,
        public_base_url=config.output.public_base_url,
        snapshot_url_prefix=config.output.snapshot_url_prefix,
        gateway_path_prefix=config.output.gateway_path_prefix,
        area_registry=area_registry,
        # camera_id → floor, so the identity gate can skip ReID/plate
        # matching on ground-floor cameras by floor rather than a hardcoded
        # camera-id set.
        camera_floors={c.id: c.floor for c in config.cameras},
        # Best-effort DB handle for clearing a stale slot's plate binding
        # after an ANPR eviction (_clear_slot_db_binding). Same handle the
        # engine uses below.
        db_manager=db,
    )

    # Single-process async engine (Phase 2 migration): one process feeds ALL
    # cameras through one OpenVINO AsyncInferQueue instead of the per-group
    # serial loops. It needs the async detector core, which TrackedDetector
    # selects at construction from VA_INFER — so force it on here, before the
    # engine (and its detector) is built.
    if _single_process and os.environ.get("VA_INFER", "").strip().lower() != "async":
        os.environ["VA_INFER"] = "async"
        print("[INFO] VA_SINGLE_PROCESS=1 → forcing VA_INFER=async")

    # Entry V2 has exactly one in-process coordinator. Both the HTTP transport
    # and VA's RTSP zone fallback feed this same bounded state machine; building
    # one per surface would split pending attempts from their local crossings.
    from src.entry.runtime import build_entry_coordinator

    entry_coordinator = build_entry_coordinator(
        registry,
        image_dir=config.output.snapshot_base_dir,
    )
    engine = ParkingEngine(
        config,
        vehicle_registry=registry,
        db_manager=db,
        entry_coordinator=entry_coordinator,
    )

    # --- Start API server if requested ---
    if args.api:
        start_api_server(
            engine,
            registry,
            port=args.port,
            entry_coordinator=entry_coordinator,
        )

    # --- Decide which mode to run ---
    if args.video:
        # Legacy single-video mode
        if args.fps:
            config.video_target_fps = args.fps
        engine.run_single_camera(
            video_source=args.video,
            slots_file=config.slots_file,
        )

    elif args.camera:
        # Single camera from multi-camera config
        cam_entry = None
        for c in config.cameras:
            if c.id == args.camera:
                cam_entry = c
                break

        if cam_entry is None:
            print(f"[ERROR] Camera '{args.camera}' not found in config.")
            print(f"[HINT] Available cameras: {[c.id for c in config.cameras]}")
            sys.exit(1)

        # Build RTSP URL and run as single camera
        channel = config.processing.stream_channel
        rtsp_url = (
            f"rtsp://{cam_entry.user}:{cam_entry.password}@{cam_entry.ip}:554"
            f"/Streaming/Channels/{channel}"
        )
        config.output.show_camera = args.camera
        if args.fps:
            config.video_target_fps = args.fps
        else:
            config.video_target_fps = config.processing.target_fps_per_camera

        engine.run_single_camera(
            video_source=rtsp_url,
            slots_file=cam_entry.slots_file,
            camera_id=cam_entry.id,
            floor=cam_entry.floor,
        )

    else:
        # Multi-camera mode (default). VA_SINGLE_PROCESS=1 switches to the
        # single-process async loop (one AsyncInferQueue for every camera).
        if _single_process:
            engine.run_single_process()
        else:
            engine.run_multi_camera()


if __name__ == "__main__":
    main()
