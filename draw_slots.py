"""
draw_slots.py — Interactive parking slot polygon drawing tool (v2).

Features:
  - Custom slot naming (type a name when finishing a slot)
  - Edit existing slots: click near a point to move it
  - Remove individual slots by clicking on them
  - Camera selection from config.yaml

Usage:
    python draw_slots.py --camera CAM_04              # Single camera from config
    python draw_slots.py --camera all                  # All cameras sequentially
    python draw_slots.py --image snapshot.jpg          # From image file
    python draw_slots.py --rtsp "rtsp://..."           # Manual RTSP URL

Controls:
    Left Click   — Add a polygon point (or grab a point in EDIT mode)
    Right Click  — Finish current slot → prompts for name in terminal
    'u'          — Undo last point
    'e'          — Toggle EDIT mode (drag existing points)
    'r'          — Toggle REMOVE mode (click a slot to delete it)
    'n'          — Rename a slot (click a slot, then type new name)
    's'          — Save and quit
    'q' / ESC    — Quit without saving
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

from src.config import load_config


class SlotDrawer:
    """Interactive polygon drawing tool with edit, remove, and rename support."""

    # Modes
    MODE_DRAW = "DRAW"
    MODE_EDIT = "EDIT"
    MODE_REMOVE = "REMOVE"
    MODE_RENAME = "RENAME"

    def __init__(self, image: np.ndarray, existing_slots: list = None, camera_label: str = ""):
        self.image = image.copy()
        self.display = image.copy()
        self.slots = existing_slots or []
        self.current_points = []
        self.camera_label = camera_label
        self.window_name = f"Draw Slots — {camera_label}" if camera_label else "Draw Parking Slots"

        # Mode state
        self.mode = self.MODE_DRAW

        # Edit mode state
        self._drag_slot_idx = -1
        self._drag_point_idx = -1
        self._is_dragging = False

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events based on current mode."""

        if self.mode == self.MODE_DRAW:
            self._handle_draw(event, x, y)
        elif self.mode == self.MODE_EDIT:
            self._handle_edit(event, x, y)
        elif self.mode == self.MODE_REMOVE:
            self._handle_remove(event, x, y)
        elif self.mode == self.MODE_RENAME:
            self._handle_rename(event, x, y)

    def _handle_draw(self, event, x, y):
        """Draw mode: left click adds points, right click finishes polygon."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append([x, y])
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_points) >= 3:
                # Prompt for slot name in terminal
                default_name = self._next_default_id()
                print(f"  Enter slot name (default: {default_name}): ", end="", flush=True)
                user_input = input().strip()
                slot_id = user_input if user_input else default_name

                self.slots.append({
                    "id": slot_id,
                    "polygon": self.current_points.copy(),
                    "label": f"Slot {slot_id}",
                })
                print(f"  [OK] Slot '{slot_id}' defined with {len(self.current_points)} points")
                self.current_points = []
                self._redraw()
            else:
                print("  [WARN] Need at least 3 points.")

    def _handle_edit(self, event, x, y):
        """Edit mode: drag existing polygon points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find nearest point across all slots
            self._drag_slot_idx, self._drag_point_idx = self._find_nearest_point(x, y, threshold=20)
            if self._drag_slot_idx >= 0:
                self._is_dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self._is_dragging:
            self.slots[self._drag_slot_idx]["polygon"][self._drag_point_idx] = [x, y]
            self._redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            self._is_dragging = False

    def _handle_remove(self, event, x, y):
        """Remove mode: click inside a slot polygon to delete it."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, slot in enumerate(self.slots):
                pts = np.array(slot["polygon"], dtype=np.float32)
                if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                    removed = self.slots.pop(i)
                    print(f"  [REMOVED] Slot '{removed['id']}'")
                    self._redraw()
                    return
            print("  [INFO] Click inside a slot polygon to remove it.")

    def _handle_rename(self, event, x, y):
        """Rename mode: click inside a slot polygon, then type new name."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for slot in self.slots:
                pts = np.array(slot["polygon"], dtype=np.float32)
                if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                    old_name = slot["id"]
                    print(f"  Rename '{old_name}' to: ", end="", flush=True)
                    new_name = input().strip()
                    if new_name:
                        slot["id"] = new_name
                        slot["label"] = f"Slot {new_name}"
                        print(f"  [RENAMED] '{old_name}' → '{new_name}'")
                    else:
                        print(f"  [SKIP] Name unchanged.")
                    self._redraw()
                    return
            print("  [INFO] Click inside a slot polygon to rename it.")

    def _find_nearest_point(self, x, y, threshold=20):
        """Find the nearest polygon point to (x, y) within threshold pixels."""
        best_dist = threshold
        best_slot = -1
        best_pt = -1
        for si, slot in enumerate(self.slots):
            for pi, pt in enumerate(slot["polygon"]):
                dist = math.hypot(pt[0] - x, pt[1] - y)
                if dist < best_dist:
                    best_dist = dist
                    best_slot = si
                    best_pt = pi
        return best_slot, best_pt

    def _next_default_id(self) -> str:
        """Generate next auto-incrementing slot ID."""
        existing_ids = {s["id"] for s in self.slots}
        n = 1
        while True:
            row = chr(ord('A') + (n - 1) // 10)
            col = ((n - 1) % 10) + 1
            candidate = f"{row}{col}"
            if candidate not in existing_ids:
                return candidate
            n += 1

    def _redraw(self):
        self.display = self.image.copy()

        # Camera label
        if self.camera_label:
            cv2.putText(self.display, self.camera_label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Mode indicator
        mode_colors = {
            self.MODE_DRAW: (0, 255, 0),
            self.MODE_EDIT: (0, 255, 255),
            self.MODE_REMOVE: (0, 0, 255),
            self.MODE_RENAME: (255, 200, 0),
        }
        mode_color = mode_colors.get(self.mode, (255, 255, 255))
        cv2.putText(self.display, f"Mode: {self.mode}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

        # Completed slots
        for slot in self.slots:
            pts = np.array(slot["polygon"], dtype=np.int32)
            cv2.polylines(self.display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw vertex circles in edit mode
            if self.mode == self.MODE_EDIT:
                for pt in slot["polygon"]:
                    cv2.circle(self.display, tuple(pt), 6, (0, 255, 255), -1)

            # Label at centroid
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(self.display, slot["id"], (cx - 15, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Current points (draw mode)
        if self.mode == self.MODE_DRAW:
            for i, pt in enumerate(self.current_points):
                cv2.circle(self.display, tuple(pt), 5, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(self.display, tuple(self.current_points[i - 1]),
                             tuple(pt), (0, 0, 255), 2)
            if len(self.current_points) >= 2:
                cv2.line(self.display, tuple(self.current_points[-1]),
                         tuple(self.current_points[0]), (0, 0, 255), 1)

        # Info bar
        h = self.display.shape[0]
        info = f"Slots: {len(self.slots)}"
        if self.mode == self.MODE_DRAW:
            info += f" | Points: {len(self.current_points)}"
        keys = "LClick=point RClick=finish | e=Edit r=Remove n=Rename | s=Save q=Quit"
        cv2.putText(self.display, info, (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(self.display, keys, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        cv2.imshow(self.window_name, self.display)

    def run(self) -> list:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print(f"\n  Modes: [DRAW] e=Edit r=Remove n=Rename | s=Save q=Quit\n")
        self._redraw()

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("  [INFO] Quit without saving.")
                cv2.destroyAllWindows()
                return None
            elif key == ord('s'):
                cv2.destroyAllWindows()
                return self.slots
            elif key == ord('u'):
                if self.mode == self.MODE_DRAW and self.current_points:
                    self.current_points.pop()
                    self._redraw()
            elif key == ord('e'):
                self.mode = self.MODE_EDIT if self.mode != self.MODE_EDIT else self.MODE_DRAW
                print(f"  [MODE] → {self.mode}")
                self._redraw()
            elif key == ord('r'):
                self.mode = self.MODE_REMOVE if self.mode != self.MODE_REMOVE else self.MODE_DRAW
                print(f"  [MODE] → {self.mode}")
                self._redraw()
            elif key == ord('n'):
                self.mode = self.MODE_RENAME if self.mode != self.MODE_RENAME else self.MODE_DRAW
                print(f"  [MODE] → {self.mode}")
                self._redraw()
            elif key == ord('d'):
                # Delete last slot (quick shortcut)
                if self.slots:
                    removed = self.slots.pop()
                    print(f"  [DELETE] Removed last slot '{removed['id']}'")
                    self._redraw()


def capture_frame(rtsp_url: str) -> np.ndarray:
    """Capture a single frame from an RTSP stream."""
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {rtsp_url}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame from: {rtsp_url}")
    return frame


def process_camera(cam_id: str, rtsp_url: str, slots_file: str, camera_label: str):
    """Run the slot drawing tool for a single camera."""
    print(f"\n{'='*60}")
    print(f"  Camera: {camera_label}")
    print(f"  Slots file: {slots_file}")
    print(f"{'='*60}")

    print(f"  Capturing frame from {cam_id}...")
    try:
        image = capture_frame(rtsp_url)
        print(f"  Frame captured: {image.shape[1]}x{image.shape[0]}")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return

    existing = []
    if os.path.exists(slots_file):
        with open(slots_file, "r") as f:
            existing = json.load(f)
        print(f"  Loaded {len(existing)} existing slots")
        resp = input("  Keep existing slots? (y/n): ").strip().lower()
        if resp != 'y':
            existing = []

    drawer = SlotDrawer(image, existing_slots=existing, camera_label=camera_label)
    result = drawer.run()

    if result is not None:
        os.makedirs(os.path.dirname(slots_file) or ".", exist_ok=True)
        with open(slots_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  [SAVED] {len(result)} slots → '{slots_file}'")
    else:
        print(f"  [SKIPPED] No changes saved for {cam_id}")


def main():
    parser = argparse.ArgumentParser(description="Interactive parking slot polygon drawer (v2).")
    parser.add_argument("--camera", type=str, default=None,
                        help="Camera ID from config (e.g., CAM_04) or 'all' for all cameras.")
    parser.add_argument("--image", type=str, default=None, help="Path to snapshot image.")
    parser.add_argument("--rtsp", type=str, default=None, help="RTSP URL to capture a live frame.")
    parser.add_argument("--output", type=str, default="parking_slots.json", help="Output JSON file.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path.")
    args = parser.parse_args()

    if args.camera:
        config = load_config(args.config)
        channel = config.processing.stream_channel

        if args.camera.lower() == "all":
            for cam in config.cameras:
                rtsp_url = f"rtsp://{cam.user}:{cam.password}@{cam.ip}:554/Streaming/Channels/{channel}"
                label = f"{cam.id} — {cam.name} ({cam.floor})"
                process_camera(cam.id, rtsp_url, cam.slots_file, label)
        else:
            cam_entry = None
            for c in config.cameras:
                if c.id == args.camera:
                    cam_entry = c
                    break
            if cam_entry is None:
                print(f"[ERROR] Camera '{args.camera}' not found.")
                print(f"Available: {[c.id for c in config.cameras]}")
                sys.exit(1)

            rtsp_url = f"rtsp://{cam_entry.user}:{cam_entry.password}@{cam_entry.ip}:554/Streaming/Channels/{channel}"
            label = f"{cam_entry.id} — {cam_entry.name} ({cam_entry.floor})"
            process_camera(cam_entry.id, rtsp_url, cam_entry.slots_file, label)

    elif args.rtsp or args.image:
        if args.rtsp:
            image = capture_frame(args.rtsp)
        else:
            image = cv2.imread(args.image)
            if image is None:
                print(f"[ERROR] Cannot read: {args.image}")
                sys.exit(1)

        existing = []
        if os.path.exists(args.output):
            with open(args.output, "r") as f:
                existing = json.load(f)
            resp = input(f"Keep {len(existing)} existing slots? (y/n): ").strip().lower()
            if resp != 'y':
                existing = []

        drawer = SlotDrawer(image, existing_slots=existing)
        result = drawer.run()

        if result is not None:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"\n[SAVED] {len(result)} slots → '{args.output}'")
    else:
        print("Usage:")
        print("  python draw_slots.py --camera CAM_04       # From config")
        print("  python draw_slots.py --camera all           # All cameras")
        print("  python draw_slots.py --rtsp 'rtsp://...'    # Manual RTSP")
        print("  python draw_slots.py --image snapshot.jpg   # From image")


if __name__ == "__main__":
    main()
