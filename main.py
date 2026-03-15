"""
main.py — Entry point for the Damanat PMS Video Analytics system.

Supports two modes:
  1. Multi-camera mode (default): processes all cameras from config.yaml
  2. Single-camera mode: --video flag for testing with one stream

Usage:
    python main.py                                     # Multi-camera mode
    python main.py --camera CAM_04 --show              # Single camera with visualization
    python main.py --video sample.mp4 --show           # Legacy single-file mode
    python main.py --show --show-camera CAM_04         # Multi-camera, visualize one

Press 'q' in the visualization window to quit (if --show is used).
Press Ctrl+C to stop at any time.
"""

import argparse
import os
import sys

from src.config import load_config
from src.core.engine import ParkingEngine


def main():
    parser = argparse.ArgumentParser(
        description="Damanat PMS Video Analytics — CPU-optimized parking slot detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                  # All cameras, round-robin
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
    args = parser.parse_args()

    # --- Load configuration ---
    print("=" * 60)
    print("  Damanat PMS Video Analytics")
    print("  CPU-Optimized Parking Management System")
    print("=" * 60)

    config = load_config(args.config)

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

    engine = ParkingEngine(config)

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
