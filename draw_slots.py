"""
draw_slots.py - Interactive parking slot polygon drawing tool (v2).

Features:
  - Custom slot naming (type a name when finishing a slot)
  - Edit existing slots: click near a point to move it
  - Remove individual slots by clicking on them
  - Camera selection from config.yaml
  - DB-backed storage for per-camera slot polygons

Usage:
    python draw_slots.py --camera CAM_04              # Single camera from config
    python draw_slots.py --camera all                 # All cameras sequentially
    python draw_slots.py --image snapshot.jpg         # From image file
    python draw_slots.py --rtsp "rtsp://..."          # Manual RTSP URL

Controls:
    Left Click   - Add a polygon point (or grab a point in EDIT mode)
    Right Click  - Finish current slot -> prompts for name in terminal
    'u'          - Undo last point
    'b'          - Toggle BOUNDARY mode (draw an area-to-area crossing band;
                   on finish, prompts for area_from / area_to)
    'e'          - Toggle EDIT mode (drag existing points)
    'r'          - Toggle REMOVE mode (click a slot to delete it)
    'n'          - Rename a slot (click a slot, then type new name)
    's'          - Save and quit
    'q' / ESC    - Quit without saving
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

from src.config import AppConfig, load_config
from src.database import init_db
from src.services.config_service import sync_app_config_from_db, sync_areas_from_db
from src.services.parking_service import (
    SLOT_TYPE_PARKING,
    SLOT_TYPE_SPECIAL_ZONE,
    load_camera_slots,
    sync_camera_boundaries,
    sync_camera_slot_definitions,
)


class SlotDrawer:
    """Interactive polygon drawing tool with edit, remove, and rename support."""

    MODE_DRAW = "DRAW"
    MODE_EDIT = "EDIT"
    MODE_REMOVE = "REMOVE"
    MODE_RENAME = "RENAME"
    MODE_BOUNDARY = "BOUNDARY"

    def __init__(self, image: np.ndarray, existing_slots: list = None, camera_label: str = "",
                 valid_areas=None):
        self.image = image.copy()
        self.display = image.copy()
        self.slots = existing_slots or []
        self.current_points = []
        self.camera_label = camera_label
        self.window_name = f"Draw Slots - {camera_label}" if camera_label else "Draw Parking Slots"
        # Known area_ids for boundary validation. Empty = skip validation.
        self.valid_areas = set(valid_areas or [])

        self.mode = self.MODE_DRAW
        self._drag_slot_idx = -1
        self._drag_point_idx = -1
        self._is_dragging = False

    def mouse_callback(self, event, x, y, flags, param):
        if self.mode == self.MODE_DRAW:
            self._handle_draw(event, x, y)
        elif self.mode == self.MODE_BOUNDARY:
            self._handle_boundary(event, x, y)
        elif self.mode == self.MODE_EDIT:
            self._handle_edit(event, x, y)
        elif self.mode == self.MODE_REMOVE:
            self._handle_remove(event, x, y)
        elif self.mode == self.MODE_RENAME:
            self._handle_rename(event, x, y)

    def _handle_draw(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append([x, y])
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_points) < 3:
                print("  [WARN] Need at least 3 points.")
                return

            default_name = self._next_default_id()
            print(f"  Enter slot name (default: {default_name}): ", end="", flush=True)
            user_input = input().strip()
            slot_id = user_input if user_input else default_name

            self.slots.append(
                {
                    "id": slot_id,
                    "polygon": self.current_points.copy(),
                    "label": slot_id,
                }
            )
            print(f"  [OK] Slot '{slot_id}' defined with {len(self.current_points)} points")
            self.current_points = []
            self._redraw()

    def _ask_area(self, role: str):
        """Prompt for an area_id, validating against ``self.valid_areas`` when
        known. Re-asks on an unknown area; blank input cancels the boundary.
        Returns the area_id, or None to cancel."""
        while True:
            print(f"  Boundary {role}: ", end="", flush=True)
            val = input().strip()
            if not val:
                print("  [CANCEL] Boundary discarded (blank area).")
                return None
            if not self.valid_areas or val in self.valid_areas:
                return val
            print(
                f"  [WARN] '{val}' is not a known area. "
                f"Valid areas: {', '.join(sorted(self.valid_areas))}"
            )

    def _handle_boundary(self, event, x, y):
        """Draw an area-to-area boundary polygon (zoning crossing zone).

        Same click/finish UX as a slot, but on finish prompts for the two
        areas the boundary connects (validated against the known areas) and
        tags the entry as a boundary."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append([x, y])
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_points) < 3:
                print("  [WARN] Need at least 3 points for a boundary band.")
                return

            default_name = f"boundary_{sum(1 for s in self.slots if s.get('type') == 'boundary') + 1}"
            print(f"  Enter boundary name (default: {default_name}): ", end="", flush=True)
            name = input().strip() or default_name

            area_from = self._ask_area("FROM area (area_from)")
            if area_from is None:
                self.current_points = []
                self._redraw()
                return
            area_to = self._ask_area("TO area   (area_to)")
            if area_to is None:
                self.current_points = []
                self._redraw()
                return
            if area_from == area_to:
                print("  [WARN] area_from and area_to are the same — boundary discarded.")
                self.current_points = []
                self._redraw()
                return

            self.slots.append(
                {
                    "id": name,
                    "polygon": self.current_points.copy(),
                    "label": name,
                    "type": "boundary",
                    "area_from": area_from,
                    "area_to": area_to,
                }
            )
            print(
                f"  [OK] Boundary '{name}' defined: {area_from} -> {area_to} "
                f"({len(self.current_points)} points)"
            )
            self.current_points = []
            self._redraw()

    def _handle_edit(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_slot_idx, self._drag_point_idx = self._find_nearest_point(x, y, threshold=20)
            if self._drag_slot_idx >= 0:
                self._is_dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self._is_dragging:
            self.slots[self._drag_slot_idx]["polygon"][self._drag_point_idx] = [x, y]
            self._redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            self._is_dragging = False

    def _handle_remove(self, event, x, y):
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
        if event == cv2.EVENT_LBUTTONDOWN:
            for slot in self.slots:
                pts = np.array(slot["polygon"], dtype=np.float32)
                if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                    old_name = slot["id"]
                    print(f"  Rename '{old_name}' to: ", end="", flush=True)
                    new_name = input().strip()
                    if new_name:
                        slot["id"] = new_name
                        slot["label"] = new_name
                        print(f"  [RENAMED] '{old_name}' -> '{new_name}'")
                    else:
                        print("  [SKIP] Name unchanged.")
                    self._redraw()
                    return
            print("  [INFO] Click inside a slot polygon to rename it.")

    def _find_nearest_point(self, x, y, threshold=20):
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
        existing_ids = {s["id"] for s in self.slots}
        n = 1
        while True:
            row = chr(ord("A") + (n - 1) // 10)
            col = ((n - 1) % 10) + 1
            candidate = f"{row}{col}"
            if candidate not in existing_ids:
                return candidate
            n += 1

    def _redraw(self):
        self.display = self.image.copy()

        if self.camera_label:
            cv2.putText(
                self.display,
                self.camera_label,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        mode_colors = {
            self.MODE_DRAW: (0, 255, 0),
            self.MODE_BOUNDARY: (255, 0, 255),
            self.MODE_EDIT: (0, 255, 255),
            self.MODE_REMOVE: (0, 0, 255),
            self.MODE_RENAME: (255, 200, 0),
        }
        cv2.putText(
            self.display,
            f"Mode: {self.mode}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            mode_colors.get(self.mode, (255, 255, 255)),
            2,
        )

        for slot in self.slots:
            is_boundary = slot.get("type") == "boundary"
            color = (255, 0, 255) if is_boundary else (0, 255, 0)
            pts = np.array(slot["polygon"], dtype=np.int32)
            cv2.polylines(self.display, [pts], isClosed=True, color=color, thickness=2)

            if self.mode == self.MODE_EDIT:
                for pt in slot["polygon"]:
                    cv2.circle(self.display, tuple(pt), 6, (0, 255, 255), -1)

            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            label = slot["id"]
            if is_boundary:
                label = f"{slot['id']} [{slot.get('area_from','?')}->{slot.get('area_to','?')}]"
            cv2.putText(
                self.display,
                label,
                (cx - 15, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        if self.mode in (self.MODE_DRAW, self.MODE_BOUNDARY):
            # In-progress polygon preview. Magenta for boundaries, red for slots.
            # Markers are scaled to the frame size so they stay visible even when
            # the window downscales a high-res (1080p/1440p) frame.
            pt_color = (255, 0, 255) if self.mode == self.MODE_BOUNDARY else (0, 0, 255)
            scale = max(1.0, self.display.shape[1] / 1280.0)
            radius = int(round(6 * scale))
            line_w = max(2, int(round(2 * scale)))
            for i, pt in enumerate(self.current_points):
                # White outline ring + filled dot so the point reads on any background.
                cv2.circle(self.display, tuple(pt), radius + 2, (255, 255, 255), -1)
                cv2.circle(self.display, tuple(pt), radius, pt_color, -1)
                if i > 0:
                    cv2.line(
                        self.display,
                        tuple(self.current_points[i - 1]),
                        tuple(pt),
                        pt_color,
                        line_w,
                    )
            if len(self.current_points) >= 2:
                cv2.line(
                    self.display,
                    tuple(self.current_points[-1]),
                    tuple(self.current_points[0]),
                    pt_color,
                    max(1, line_w // 2),
                )

        h = self.display.shape[0]
        info = f"Slots: {len(self.slots)}"
        if self.mode in (self.MODE_DRAW, self.MODE_BOUNDARY):
            info += f" | Points: {len(self.current_points)}"
        keys = "LClick=point RClick=finish | b=Boundary e=Edit r=Remove n=Rename | s=Save q=Quit"
        cv2.putText(
            self.display,
            info,
            (10, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            self.display,
            keys,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 200, 200),
            1,
        )

        cv2.imshow(self.window_name, self.display)

    def run(self) -> list:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n  Modes: [DRAW] e=Edit r=Remove n=Rename | s=Save q=Quit\n")
        self._redraw()

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("  [INFO] Quit without saving.")
                cv2.destroyAllWindows()
                return None
            if key == ord("s"):
                cv2.destroyAllWindows()
                return self.slots
            if key == ord("u"):
                if self.mode in (self.MODE_DRAW, self.MODE_BOUNDARY) and self.current_points:
                    self.current_points.pop()
                    self._redraw()
            elif key == ord("e"):
                self.mode = self.MODE_EDIT if self.mode != self.MODE_EDIT else self.MODE_DRAW
                print(f"  [MODE] -> {self.mode}")
                self._redraw()
            elif key == ord("r"):
                self.mode = self.MODE_REMOVE if self.mode != self.MODE_REMOVE else self.MODE_DRAW
                print(f"  [MODE] -> {self.mode}")
                self._redraw()
            elif key == ord("n"):
                self.mode = self.MODE_RENAME if self.mode != self.MODE_RENAME else self.MODE_DRAW
                print(f"  [MODE] -> {self.mode}")
                self._redraw()
            elif key == ord("b"):
                self.mode = self.MODE_BOUNDARY if self.mode != self.MODE_BOUNDARY else self.MODE_DRAW
                print(f"  [MODE] -> {self.mode}")
                self._redraw()
            elif key == ord("d"):
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


def _runtime_slot_to_editor_entry(slot) -> dict:
    return {
        "id": slot.id,
        "polygon": [[int(x), int(y)] for x, y in list(slot.polygon.exterior.coords)[:-1]],
        "label": slot.label or slot.id,
        "zone_id": slot.zone_id,
        "zone_name": slot.zone_name,
    }


def _boundary_to_editor_entry(boundary) -> dict:
    """Editor entry for a loaded boundary zone (zoning crossing polygon)."""
    return {
        "id": boundary.id,
        "polygon": [[int(x), int(y)] for x, y in list(boundary.polygon.exterior.coords)[:-1]],
        "label": boundary.id,
        "type": "boundary",
        "area_from": boundary.area_from,
        "area_to": boundary.area_to,
    }


def process_camera(cam_id: str, rtsp_url: str, camera_label: str, config: AppConfig, db_manager):
    """Run the slot drawing tool for a single camera."""
    print(f"\n{'=' * 60}")
    print(f"  Camera: {camera_label}")
    print("  Storage: parking_slots table")
    print(f"{'=' * 60}")

    print(f"  Capturing frame from {cam_id}...")
    try:
        image = capture_frame(rtsp_url)
        print(f"  Frame captured: {image.shape[1]}x{image.shape[0]}")
    except Exception as exc:
        print(f"  [ERROR] {exc}")
        return

    existing = []
    session = db_manager.SessionLocal()
    try:
        ref_res = (
            config.processing.slot_ref_width,
            config.processing.slot_ref_height,
        )
        actual_res = (image.shape[1], image.shape[0])
        parking_slots, special_zones, _, boundaries = load_camera_slots(
            session,
            camera_id=cam_id,
            ref_resolution=ref_res,
            actual_resolution=actual_res,
        )
        existing = [
            _runtime_slot_to_editor_entry(slot)
            for slot in parking_slots + special_zones
        ]
        existing += [_boundary_to_editor_entry(b) for b in boundaries]
    finally:
        session.close()

    if existing:
        print(f"  Loaded {len(existing)} existing definitions from database")
        resp = input("  Keep existing slots? (y/n): ").strip().lower()
        if resp != "y":
            existing = []

    valid_areas = {a.area_id for a in config.areas if a.area_id}
    drawer = SlotDrawer(
        image, existing_slots=existing, camera_label=camera_label, valid_areas=valid_areas
    )
    result = drawer.run()

    if result is None:
        print(f"  [SKIPPED] No changes saved for {cam_id}")
        return

    ref_w = config.processing.slot_ref_width
    ref_h = config.processing.slot_ref_height
    act_w = image.shape[1]
    act_h = image.shape[0]

    if ref_w > 0 and ref_h > 0 and (ref_w != act_w or ref_h != act_h):
        sx = ref_w / act_w
        sy = ref_h / act_h
        print(f"  [INFO] Scaling polygons back to reference resolution ({ref_w}x{ref_h}) for saving...")
        for slot in result:
            slot["polygon"] = [[round(p[0] * sx, 1), round(p[1] * sy, 1)] for p in slot["polygon"]]

    floor = next((c.floor for c in config.cameras if c.id == cam_id), "")
    boundary_entries = [e for e in result if e.get("type") == "boundary"]
    slot_entries = [e for e in result if e.get("type") != "boundary"]

    session = db_manager.SessionLocal()
    try:
        sync_camera_slot_definitions(
            db=session,
            camera_id=cam_id,
            floor=floor,
            slot_entries=slot_entries,
            managed_slot_types=(SLOT_TYPE_PARKING, SLOT_TYPE_SPECIAL_ZONE),
            default_zone_id=cam_id,
            default_zone_name=cam_id,
        )
        # Boundaries live in their own table (zoning).
        sync_camera_boundaries(
            db=session,
            camera_id=cam_id,
            floor=floor,
            boundary_entries=boundary_entries,
        )
    finally:
        session.close()

    print(
        f"  [SAVED] {len(slot_entries)} slot(s) -> parking_slots, "
        f"{len(boundary_entries)} boundary(ies) -> boundaries ({cam_id})"
    )


def main():
    parser = argparse.ArgumentParser(description="Interactive parking slot polygon drawer (v2).")
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Camera ID from config (e.g., CAM_04) or 'all' for all cameras.",
    )
    parser.add_argument("--image", type=str, default=None, help="Path to snapshot image.")
    parser.add_argument("--rtsp", type=str, default=None, help="RTSP URL to capture a live frame.")
    parser.add_argument("--output", type=str, default="parking_slots.json", help="Output JSON file.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path.")
    args = parser.parse_args()

    if args.camera:
        config = load_config(args.config)
        db = None

        try:
            db = init_db(config.database.url)
            session = db.SessionLocal()
            sync_app_config_from_db(session, config)
            sync_areas_from_db(session, config)  # so boundary validation sees DB areas
            session.close()
        except Exception as exc:
            print(f"[WARN] Could not sync with database: {exc}. Using YAML/defaults.")
        if db is None:
            print("[ERROR] Database connection is required for --camera mode.")
            sys.exit(1)

        channel = config.processing.stream_channel

        if args.camera.lower() == "all":
            for cam in config.cameras:
                rtsp_url = (
                    f"rtsp://{cam.user}:{cam.password}@{cam.ip}:554/Streaming/Channels/{channel}"
                )
                label = f"{cam.id} - {cam.name} ({cam.floor})"
                process_camera(cam.id, rtsp_url, label, config, db)
        else:
            cam_entry = None
            for camera in config.cameras:
                if camera.id == args.camera:
                    cam_entry = camera
                    break
            if cam_entry is None:
                print(f"[ERROR] Camera '{args.camera}' not found.")
                print(f"Available: {[c.id for c in config.cameras]}")
                sys.exit(1)

            rtsp_url = (
                f"rtsp://{cam_entry.user}:{cam_entry.password}@{cam_entry.ip}:554/Streaming/Channels/{channel}"
            )
            label = f"{cam_entry.id} - {cam_entry.name} ({cam_entry.floor})"
            process_camera(cam_entry.id, rtsp_url, label, config, db)

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
            with open(args.output, "r", encoding="utf-8") as f:
                existing = json.load(f)
            resp = input(f"Keep {len(existing)} existing slots? (y/n): ").strip().lower()
            if resp != "y":
                existing = []

        drawer = SlotDrawer(image, existing_slots=existing)
        result = drawer.run()

        if result is not None:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"\n[SAVED] {len(result)} slots -> '{args.output}'")
    else:
        print("Usage:")
        print("  python draw_slots.py --camera CAM_04       # From config")
        print("  python draw_slots.py --camera all          # All cameras")
        print("  python draw_slots.py --rtsp 'rtsp://...'   # Manual RTSP")
        print("  python draw_slots.py --image snapshot.jpg  # From image")


if __name__ == "__main__":
    main()
