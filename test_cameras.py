"""
test_cameras.py — Quick connectivity test for all cameras.

Tests each camera's RTSP stream and reports status.

Usage:
    python test_cameras.py
    python test_cameras.py --config config.yaml
"""

import argparse
import sys
import cv2

from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Test all camera connections.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--channel", type=int, default=None,
                        help="Override stream channel (101=4K, 102=720p)")
    args = parser.parse_args()

    config = load_config(args.config)
    channel = args.channel or config.processing.stream_channel

    if not config.cameras:
        print("[ERROR] No cameras defined in config.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Camera Connectivity Test — {len(config.cameras)} cameras")
    print(f"  Stream channel: {channel}")
    print(f"{'='*60}\n")

    results = []

    for cam in config.cameras:
        rtsp_url = (
            f"rtsp://{cam.user}:{cam.password}@{cam.ip}:554"
            f"/Streaming/Channels/{channel}"
        )

        print(f"  [{cam.id}] {cam.name} ({cam.floor}) — {cam.ip} ... ", end="", flush=True)

        try:
            cap = cv2.VideoCapture(rtsp_url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    print(f"OK ({w}x{h})")
                    results.append((cam.id, True, f"{w}x{h}"))
                else:
                    print("OPEN but no frame")
                    results.append((cam.id, False, "no frame"))
            else:
                print("FAILED")
                results.append((cam.id, False, "cannot open"))
            cap.release()
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((cam.id, False, str(e)))

    # Summary
    ok = sum(1 for _, success, _ in results if success)
    fail = len(results) - ok

    print(f"\n{'='*60}")
    print(f"  Results: {ok} OK, {fail} FAILED out of {len(results)}")
    print(f"{'='*60}")

    if fail > 0:
        print("\n  Failed cameras:")
        for cam_id, success, info in results:
            if not success:
                print(f"    - {cam_id}: {info}")

    print()


if __name__ == "__main__":
    main()
