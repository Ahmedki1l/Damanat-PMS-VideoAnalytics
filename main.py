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

from src.config import load_config
from src.core.engine import ParkingEngine
from src.database import init_db

def start_api_server(engine, registry, host="0.0.0.0", port=8000):
    """Start the FastAPI server in a background thread."""
    import uvicorn
    from src.api import create_app

    def get_slot_statuses():
        """Callback for the API to get current slot statuses."""
        all_statuses = []
        for cam_id, pipeline in engine.pipelines.items():
            for sm in pipeline.state_machines.values():
                status = sm.get_status()
                status["camera_id"] = cam_id
                status["floor"] = pipeline.floor
                all_statuses.append(status)
        return all_statuses

    def get_camera_frame(cam_id: str):
        if hasattr(engine, "cam_manager") and engine.cam_manager:
            return engine.cam_manager.read_camera(cam_id)
        return False, None

    def get_park_entry_crop(cam_id: str):
        success, frame = get_camera_frame(cam_id)
        if not success or frame is None:
            return False, None
        zones = getattr(engine, "special_zones", {}).get(cam_id, {})
        for zone_id, zone in zones.items():
            if "Park_Entry" in zone_id:
                minx, miny, maxx, maxy = zone.polygon.bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, int(minx)), max(0, int(miny))
                x2, y2 = min(w, int(maxx)), min(h, int(maxy))
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    return True, crop
        return True, frame

    app = create_app(
        vehicle_registry=registry,
        get_slot_statuses=get_slot_statuses,
        get_camera_frame=get_camera_frame,
        get_park_entry_crop=get_park_entry_crop,
    )

    # Include routers
    from src.routers.parking_router import router as parking_router
    from src.routers.slot_status_router import router as slot_status_router
    from src.routers.intrusion_router import router as intrusion_router
    app.include_router(parking_router)
    app.include_router(slot_status_router)
    app.include_router(intrusion_router)

    def run_server():
        uvicorn.run(app, host=host, port=port, log_level="info")

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    print(f"[INFO] API server started at http://{host}:{port}")
    print(f"[INFO] Docs: http://{host}:{port}/docs\n")
    return thread


def main():
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
    args = parser.parse_args()

    # --- Load configuration ---
    print("=" * 60)
    print("  Damanat PMS Video Analytics")
    print("  CPU-Optimized Parking Management System")
    print("=" * 60)

    config = load_config(args.config)
    db = init_db(config.database.url)
    db.create_tables()

    # Apply CLI overrides
    if args.show:
        config.output.show_video = True
    if args.show_camera:
        config.output.show_camera = args.show_camera

    # Validate model exists
    if not os.path.exists(config.detector.model_path):
        print(f"\n[ERROR] Model file not found: '{config.detector.model_path}'")
        print(f"[HINT] Run 'python setup_model.py' first to download the model.")
        sys.exit(1)

    # --- Initialize vehicle registry (shared between engine and API) ---
    registry = None
    if args.api:
        from src.vehicle_registry import VehicleRegistry
        registry = VehicleRegistry()

    engine = ParkingEngine(config, vehicle_registry=registry, db_manager=db)

    # --- Start API server if requested ---
    if args.api:
        start_api_server(engine, registry, port=args.port)

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
        )

    else:
        # Multi-camera mode (default)
        engine.run_multi_camera()


if __name__ == "__main__":
    main()
