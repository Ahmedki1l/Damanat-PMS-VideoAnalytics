import os

import cv2
import numpy as np

from src.config import load_config
from src.database import init_db
from src.services.parking_service import load_camera_slots, save_camera_roi

# Click within this many pixels of an existing point to grab it instead of
# adding a new one.
GRAB_RADIUS = 12


def build_rtsp_url(camera_config, channel=102):
    user = camera_config["user"]
    password = camera_config["password"]
    ip = camera_config["ip"]
    return f"rtsp://{user}:{password}@{ip}:554/Streaming/Channels/{channel}"


class ROISelector:
    def __init__(self, camera_id, floor, rtsp_url, db_manager, ref_resolution):
        self.camera_id = camera_id
        self.floor = floor
        self.rtsp_url = rtsp_url
        self.db_manager = db_manager
        self.ref_resolution = ref_resolution
        self.points = []
        self._drag_idx = -1
        # Reference geometry drawn underneath the ROI so it can be aligned to the
        # existing boundaries / slots. Each entry is (int-pts array, label).
        self._ov_slots = []
        self._ov_zones = []
        self._ov_bounds = []
        self._show_overlay = True
        self.window_name = (
            f"ROI Selector - {camera_id} "
            "(click=add, drag=move, right-click=del, u=undo, c=clear, "
            "b=toggle boundaries, s=save, q=quit)"
        )

    def _nearest_point_idx(self, x, y):
        """Index of the existing point within GRAB_RADIUS of (x, y), or -1."""
        best_idx, best_d2 = -1, GRAB_RADIUS * GRAB_RADIUS
        for i, (px, py) in enumerate(self.points):
            d2 = (px - x) ** 2 + (py - y) ** 2
            if d2 <= best_d2:
                best_idx, best_d2 = i, d2
        return best_idx

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Grab an existing point if the click landed on one; else add a point.
            idx = self._nearest_point_idx(x, y)
            if idx >= 0:
                self._drag_idx = idx
            else:
                self.points.append([x, y])
            self.draw()
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_idx >= 0:
            self.points[self._drag_idx] = [x, y]
            self.draw()
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag_idx = -1
        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = self._nearest_point_idx(x, y)
            if idx >= 0:
                del self.points[idx]
                self.draw()

    def _draw_overlay(self, img):
        """Draw the reference geometry (slots grey, special zones cyan,
        boundaries magenta with an area_from->area_to label) under the ROI."""
        def _poly(pts, color, thickness, label=None):
            if pts is None or len(pts) < 2:
                return
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
            if label:
                cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                cv2.putText(img, label, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, color, 1, cv2.LINE_AA)

        for pts, label in self._ov_slots:
            _poly(pts, (160, 160, 160), 1, label)
        for pts, label in self._ov_zones:
            _poly(pts, (255, 255, 0), 1, label)
        for pts, label in self._ov_bounds:
            _poly(pts, (255, 0, 255), 2, label)

    def draw(self):
        img_copy = self.img.copy()
        if self._show_overlay:
            self._draw_overlay(img_copy)
        if self.points:
            for pt in self.points:
                cv2.circle(img_copy, tuple(pt), 5, (0, 255, 0), -1)
            if len(self.points) > 1:
                cv2.polylines(
                    img_copy,
                    [np.array(self.points)],
                    isClosed=False,
                    color=(0, 255, 0),
                    thickness=2,
                )
            if len(self.points) > 2:
                cv2.line(img_copy, tuple(self.points[-1]), tuple(self.points[0]), (0, 255, 255), 1)

        cv2.imshow(self.window_name, img_copy)

    def run(self):
        print(f"[INFO] Connecting to {self.camera_id}...")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(self.rtsp_url)

        ret, self.img = cap.read()
        cap.release()

        if not ret or self.img is None:
            print(f"[ERROR] Could not grab frame from {self.camera_id}. Check RTSP URL/Connection.")
            return

        self._load_existing_geometry()

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.draw()

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                self.points = []
                self.draw()
            elif key == ord("b"):
                self._show_overlay = not self._show_overlay
                self.draw()
            elif key == ord("u"):
                if self.points:
                    self.points.pop()
                    self.draw()
            elif key == ord("s"):
                if len(self.points) >= 3:
                    self.save_roi()
                    break
                print("[WARN] Need at least 3 points to save ROI.")

        cv2.destroyAllWindows()

    @staticmethod
    def _poly_to_pts(polygon):
        """Shapely polygon -> int Nx2 pixel array (closing point dropped)."""
        coords = list(polygon.exterior.coords)[:-1]
        return np.array([[int(round(x)), int(round(y))] for x, y in coords], dtype=np.int32)

    def _load_existing_geometry(self):
        """Pre-load this camera's saved ROI as editable points, plus the existing
        slots / special zones / boundaries as a read-only reference overlay — all
        scaled from the stored reference resolution to the live frame's pixels."""
        act_h, act_w = self.img.shape[:2]
        session = self.db_manager.SessionLocal()
        try:
            slots, zones, roi_polygon, boundaries = load_camera_slots(
                session,
                camera_id=self.camera_id,
                ref_resolution=self.ref_resolution,
                actual_resolution=(act_w, act_h),
            )
        finally:
            session.close()

        self._ov_slots = [(self._poly_to_pts(s.polygon), s.label) for s in slots]
        self._ov_zones = [(self._poly_to_pts(z.polygon), z.label) for z in zones]
        self._ov_bounds = [
            (self._poly_to_pts(b.polygon), f"{b.area_from}->{b.area_to}")
            for b in boundaries
        ]
        print(
            f"[INFO] Reference overlay: {len(self._ov_slots)} slot(s), "
            f"{len(self._ov_zones)} zone(s), {len(self._ov_bounds)} boundary(ies) "
            f"— 'b' toggles."
        )

        if roi_polygon is None:
            print(f"[INFO] No existing ROI for {self.camera_id} — starting fresh.")
            return

        self.points = [list(pt) for pt in self._poly_to_pts(roi_polygon)]
        print(f"[INFO] Loaded existing ROI for {self.camera_id} ({len(self.points)} points) — edit and 's' to save.")

    def save_roi(self):
        points = [p[:] for p in self.points]
        ref_w, ref_h = self.ref_resolution
        act_h, act_w = self.img.shape[:2]
        if ref_w > 0 and ref_h > 0 and (ref_w != act_w or ref_h != act_h):
            sx = ref_w / act_w
            sy = ref_h / act_h
            points = [[round(p[0] * sx, 1), round(p[1] * sy, 1)] for p in points]

        session = self.db_manager.SessionLocal()
        try:
            save_camera_roi(
                session,
                camera_id=self.camera_id,
                floor=self.floor,
                polygon_points=points,
            )
        finally:
            session.close()

        print(f"[SUCCESS] ROI saved to parking_slots for {self.camera_id}")


if __name__ == "__main__":
    import sys

    from src.services.config_service import load_cameras_from_db

    config = load_config()
    db = init_db(config.database.url)

    # DB-first roster: replace the YAML cameras with the enabled rows from the
    # `cameras` table (decrypted passwords, ANPR cameras excluded) — the same
    # source of truth the engine uses. Falls back to YAML when the table is
    # absent/empty.
    _session = db.SessionLocal()
    try:
        load_cameras_from_db(_session, config)
    finally:
        _session.close()

    raw_config = {"cameras": [], "processing": {"stream_channel": config.processing.stream_channel}}
    for cam in config.cameras:
        raw_config["cameras"].append(
            {
                "id": cam.id,
                "name": cam.name,
                "floor": cam.floor,
                "ip": cam.ip,
                "user": cam.user,
                "password": cam.password,
            }
        )

    # ROIs are per-camera and apply to B1/B2 exactly as to the ground floor.
    # A camera id may be passed on the command line
    # (e.g. `python -m src.tools.roi_selector CAM-07`); otherwise every enabled
    # camera is listed and prompted for.
    arg_id = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    default_id = raw_config["cameras"][0]["id"] if raw_config["cameras"] else "CAM-01"

    if not arg_id:
        print("Available Cameras for ROI definition:")
        for cam in raw_config["cameras"]:
            print(f" - {cam['id']}: {cam['name']} ({cam['floor']})")

    cam_id = (
        arg_id
        or input(f"\nEnter Camera ID to configure (or press Enter for {default_id}): ").strip()
        or default_id
    )

    selected_cam = next((c for c in raw_config["cameras"] if c["id"] == cam_id), None)
    if not selected_cam:
        print(f"[ERROR] Camera {cam_id} not found in config.")
    else:
        url = build_rtsp_url(selected_cam, channel=raw_config["processing"].get("stream_channel", 102))
        selector = ROISelector(
            cam_id,
            selected_cam["floor"],
            url,
            db,
            (config.processing.slot_ref_width, config.processing.slot_ref_height),
        )
        selector.run()
