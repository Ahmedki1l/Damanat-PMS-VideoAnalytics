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
        self.window_name = (
            f"ROI Selector - {camera_id} "
            "(drag=move pt, click=add, right-click=del, u=undo, c=clear, s=save, q=quit)"
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

    def draw(self):
        img_copy = self.img.copy()
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

        self._load_existing_roi()

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

    def _load_existing_roi(self):
        """Pre-load this camera's saved ROI (if any) as editable points, scaled
        from the stored reference resolution to the live frame's pixels."""
        act_h, act_w = self.img.shape[:2]
        session = self.db_manager.SessionLocal()
        try:
            _, _, roi_polygon, _ = load_camera_slots(
                session,
                camera_id=self.camera_id,
                ref_resolution=self.ref_resolution,
                actual_resolution=(act_w, act_h),
            )
        finally:
            session.close()

        if roi_polygon is None:
            print(f"[INFO] No existing ROI for {self.camera_id} — starting fresh.")
            return

        # Drop the closing point that shapely repeats to close the ring.
        coords = list(roi_polygon.exterior.coords)[:-1]
        self.points = [[int(round(x)), int(round(y))] for x, y in coords]
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
    config = load_config()
    db = init_db(config.database.url)
    target_cams = ["CAM-01", "CAM-02"]

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

    print("Available Cameras for ROI definition:")
    for cam in raw_config["cameras"]:
        if cam["id"] in target_cams:
            print(f" - {cam['id']}: {cam['name']} ({cam['floor']})")

    cam_id = input("\nEnter Camera ID to configure (or press Enter for CAM-01): ").strip() or "CAM-01"

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
