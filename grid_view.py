"""
grid_view.py — Multi-camera live grid view.

Shows all camera feeds on a single screen in a grid layout,
with slot polygons and occupancy status overlay.

Usage:
    python grid_view.py                        # All cameras in grid
    python grid_view.py --floor B1             # Only B1 cameras
    python grid_view.py --cols 4               # 4 columns layout
    python grid_view.py --width 1920 --height 1080  # Custom window size

Controls:
    'q' / ESC   — Quit
    'f'         — Toggle fullscreen
    '1'-'9'     — Show only that camera full-screen (press again to return)
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import load_config, AppConfig
from src.camera_manager import CameraConfig, CameraStream


# Slot state colors (BGR)
SLOT_COLORS = {
    "VACANT": (0, 255, 0),       # Green
    "ENTERING": (0, 255, 255),   # Yellow
    "OCCUPIED": (0, 0, 255),     # Red
    "LEAVING": (0, 165, 255),    # Orange
}


class GridView:
    """Multi-camera grid display."""

    def __init__(
        self,
        camera_configs: List[CameraConfig],
        slots_data: Dict[str, list],
        grid_cols: int = 0,
        window_width: int = 1920,
        window_height: int = 1080,
    ):
        self.camera_configs = camera_configs
        self.slots_data = slots_data  # cam_id → list of slot dicts
        self.window_width = window_width
        self.window_height = window_height

        # Calculate grid layout
        n = len(camera_configs)
        if grid_cols > 0:
            self.cols = grid_cols
        else:
            self.cols = math.ceil(math.sqrt(n))
        self.rows = math.ceil(n / self.cols)

        # Cell size
        self.cell_w = window_width // self.cols
        self.cell_h = window_height // self.rows

        # Open streams
        self.streams: Dict[str, CameraStream] = {}
        self.cam_ids: List[str] = []
        for cc in camera_configs:
            stream = CameraStream(cc)
            self.streams[cc.id] = stream
            self.cam_ids.append(cc.id)

        # Solo view (single camera fullscreen)
        self._solo_cam: Optional[str] = None
        self._fullscreen = False

    def open_all(self) -> int:
        print(f"\n[INFO] Opening {len(self.streams)} camera streams...")
        count = 0
        for cam_id in self.cam_ids:
            if self.streams[cam_id].open():
                count += 1
        print(f"[INFO] {count}/{len(self.streams)} connected.\n")
        return count

    def run(self):
        """Main display loop."""
        window_name = "Damanat PMS — All Cameras"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)

        print(f"[INFO] Grid: {self.rows}x{self.cols} ({len(self.cam_ids)} cameras)")
        print(f"[INFO] Cell size: {self.cell_w}x{self.cell_h}")
        print(f"[INFO] Press 'q' to quit, 'f' for fullscreen\n")

        frame_count = 0
        t_start = time.time()

        try:
            while True:
                if self._solo_cam:
                    # Single camera fullscreen
                    canvas = self._render_solo()
                else:
                    # Grid view
                    canvas = self._render_grid()

                cv2.imshow(window_name, canvas)

                frame_count += 1
                if frame_count % 60 == 0:
                    elapsed = time.time() - t_start
                    fps = frame_count / elapsed
                    active = sum(1 for s in self.streams.values() if s.is_open)
                    print(f"[PERF] Grid FPS: {fps:.1f} | Active: {active}/{len(self.streams)}")

                key = cv2.waitKey(30) & 0xFF

                if key == ord('q') or key == 27:
                    break
                elif key == ord('f'):
                    self._fullscreen = not self._fullscreen
                    if self._fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                              cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                              cv2.WINDOW_NORMAL)
                elif ord('1') <= key <= ord('9'):
                    idx = key - ord('1')
                    if idx < len(self.cam_ids):
                        cam_id = self.cam_ids[idx]
                        if self._solo_cam == cam_id:
                            self._solo_cam = None  # Toggle back to grid
                        else:
                            self._solo_cam = cam_id

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted.")

        finally:
            for stream in self.streams.values():
                stream.close()
            cv2.destroyAllWindows()

    def _render_grid(self) -> np.ndarray:
        """Render all cameras in a grid layout."""
        canvas = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

        for idx, cam_id in enumerate(self.cam_ids):
            row = idx // self.cols
            col = idx % self.cols
            x_off = col * self.cell_w
            y_off = row * self.cell_h

            ret, frame = self.streams[cam_id].read()
            if ret and frame is not None:
                cell = self._prepare_cell(frame, cam_id)
            else:
                cell = self._offline_cell(cam_id)

            canvas[y_off:y_off + self.cell_h, x_off:x_off + self.cell_w] = cell

        # Grid lines
        for i in range(1, self.cols):
            x = i * self.cell_w
            cv2.line(canvas, (x, 0), (x, self.window_height), (80, 80, 80), 1)
        for i in range(1, self.rows):
            y = i * self.cell_h
            cv2.line(canvas, (0, y), (self.window_width, y), (80, 80, 80), 1)

        return canvas

    def _render_solo(self) -> np.ndarray:
        """Render a single camera fullscreen."""
        ret, frame = self.streams[self._solo_cam].read()
        if ret and frame is not None:
            # Draw slots
            self._draw_slots_on_frame(frame, self._solo_cam)
            # Label
            cc = None
            for c in self.camera_configs:
                if c.id == self._solo_cam:
                    cc = c
                    break
            label = f"{self._solo_cam} — {cc.name if cc else ''} ({cc.floor if cc else ''})"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Press same key to return to grid", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            return cv2.resize(frame, (self.window_width, self.window_height))
        else:
            return self._offline_cell(self._solo_cam, (self.window_width, self.window_height))

    def _prepare_cell(self, frame: np.ndarray, cam_id: str) -> np.ndarray:
        """Resize frame to cell size and draw overlays."""
        cell = cv2.resize(frame, (self.cell_w, self.cell_h))

        # Draw slots scaled to cell size
        self._draw_slots_on_cell(cell, cam_id, frame.shape[:2])

        # Camera label
        cc = None
        for c in self.camera_configs:
            if c.id == cam_id:
                cc = c
                break
        label = f"{cam_id}"
        if cc:
            label += f" ({cc.floor})"
        cv2.putText(cell, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return cell

    def _draw_slots_on_cell(self, cell, cam_id, orig_shape):
        """Draw slot polygons on a resized cell."""
        if cam_id not in self.slots_data:
            return

        orig_h, orig_w = orig_shape
        scale_x = self.cell_w / orig_w
        scale_y = self.cell_h / orig_h

        for slot in self.slots_data[cam_id]:
            pts = np.array(slot["polygon"], dtype=np.float32)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            pts = pts.astype(np.int32)

            color = SLOT_COLORS.get("VACANT", (0, 255, 0))
            cv2.polylines(cell, [pts], isClosed=True, color=color, thickness=1)

            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(cell, slot["id"], (cx - 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    def _draw_slots_on_frame(self, frame, cam_id):
        """Draw slot polygons on full-size frame."""
        if cam_id not in self.slots_data:
            return

        for slot in self.slots_data[cam_id]:
            pts = np.array(slot["polygon"], dtype=np.int32)
            color = SLOT_COLORS.get("VACANT", (0, 255, 0))
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(frame, slot["id"], (cx - 15, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _offline_cell(self, cam_id: str, size=None) -> np.ndarray:
        """Create an 'offline' placeholder cell."""
        w, h = size or (self.cell_w, self.cell_h)
        cell = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(cell, cam_id, (w // 4, h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
        cv2.putText(cell, "OFFLINE", (w // 4, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
        return cell


def main():
    parser = argparse.ArgumentParser(description="Multi-camera grid view.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--floor", type=str, default=None, help="Filter by floor (B1, B2)")
    parser.add_argument("--cols", type=int, default=0, help="Number of grid columns (auto if 0)")
    parser.add_argument("--width", type=int, default=1920, help="Window width")
    parser.add_argument("--height", type=int, default=1080, help="Window height")
    args = parser.parse_args()

    config = load_config(args.config)
    channel = config.processing.stream_channel

    # Filter cameras
    cameras = config.cameras
    if args.floor:
        cameras = [c for c in cameras if c.floor.upper() == args.floor.upper()]
        print(f"[INFO] Filtering to floor: {args.floor} ({len(cameras)} cameras)")

    if not cameras:
        print("[ERROR] No cameras found.")
        sys.exit(1)

    # Build camera configs with RTSP URLs
    cam_configs = []
    for cam in cameras:
        cc = CameraConfig(
            id=cam.id, name=cam.name, floor=cam.floor,
            ip=cam.ip, user=cam.user, password=cam.password,
            slots_file=cam.slots_file,
        )
        cc.build_rtsp_url(channel=channel)
        cam_configs.append(cc)

    # Load slot data for each camera
    slots_data = {}
    for cam in cameras:
        if cam.slots_file and os.path.exists(cam.slots_file):
            with open(cam.slots_file, "r") as f:
                slots_data[cam.id] = json.load(f)

    # Create and run grid view
    grid = GridView(
        camera_configs=cam_configs,
        slots_data=slots_data,
        grid_cols=args.cols,
        window_width=args.width,
        window_height=args.height,
    )
    grid.open_all()
    grid.run()


if __name__ == "__main__":
    main()
